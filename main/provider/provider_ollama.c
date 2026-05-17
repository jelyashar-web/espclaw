/*
 * ESPClaw - provider/provider_ollama.c
 * Native Ollama API provider.
 * POST /api/generate  or  /api/chat
 * Uses Ollama's native protocol instead of OpenAI-compatible layer.
 *
 * Request format (generate):
 *   {"model":"llama3.1","prompt":"...","stream":false}
 *
 * Response format (generate):
 *   {"response":"...","done":true}
 *
 * We use /api/generate because it is lighter on RAM than /api/chat
 * and fits better on ESP32.
 */
#include "provider.h"
#include "config.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_crt_bundle.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static const char *TAG = "ollama";

/* HTTP event handler — accumulate response body */
typedef struct {
    char  *buf;
    size_t buf_sz;
    size_t written;
} http_ctx_t;

static esp_err_t http_event_handler(esp_http_client_event_t *evt)
{
    http_ctx_t *ctx = (http_ctx_t *)evt->user_data;
    if (evt->event_id == HTTP_EVENT_ON_DATA && evt->data_len > 0) {
        size_t remaining = ctx->buf_sz - ctx->written - 1;
        size_t to_copy   = (evt->data_len < (int)remaining)
                           ? (size_t)evt->data_len : remaining;
        if (to_copy > 0) {
            memcpy(ctx->buf + ctx->written, evt->data, to_copy);
            ctx->written += to_copy;
            ctx->buf[ctx->written] = '\0';
        }
    }
    return ESP_OK;
}

/*
 * Extract text from Ollama /api/generate response:
 *   {"response":"...","done":true}
 */
static void extract_response(const char *json, char *out, size_t out_sz)
{
    const char *key = strstr(json, "\"response\":\"");
    if (!key) {
        strncpy(out, "[no response in Ollama output]", out_sz - 1);
        out[out_sz - 1] = '\0';
        return;
    }
    const char *start = key + strlen("\"response\":\"");
    size_t pos = 0;
    while (*start && pos < out_sz - 1) {
        if (start[0] == '\\' && start[1] == '"') {
            out[pos++] = '"';
            start += 2;
        } else if (start[0] == '\\' && start[1] == 'n') {
            out[pos++] = '\n';
            start += 2;
        } else if (start[0] == '\\' && start[1] == '\\') {
            out[pos++] = '\\';
            start += 2;
        } else if (start[0] == '"') {
            break;
        } else {
            out[pos++] = *start++;
        }
    }
    out[pos] = '\0';
}

static char s_model[64];
static char s_base_url[128];

static esp_err_t ollama_init(const char *api_key, const char *model,
                             const char *base_url)
{
    (void)api_key; /* Ollama does not require an API key by default */
    strncpy(s_model, model, sizeof(s_model) - 1);
    if (base_url && strlen(base_url) > 0)
        strncpy(s_base_url, base_url, sizeof(s_base_url) - 1);
    else
        strncpy(s_base_url, LLM_API_URL_OLLAMA, sizeof(s_base_url) - 1);
    return ESP_OK;
}

/*
 * Build a single prompt string from system_prompt + messages_json.
 * Ollama /api/generate takes a flat prompt, so we concatenate
 * system + conversation history into one prompt.
 */
/*
 * Build a single prompt string from system_prompt + messages_json.
 * Ollama /api/generate takes a flat prompt, so we concatenate
 * system + conversation history into one prompt.
 *
 * Handles OpenAI-format messages, skipping tool_calls entries where
 * content is null.
 */
static int build_prompt(const char *system_prompt,
                        const char *messages_json,
                        char *out, size_t out_sz)
{
    size_t pos = 0;

    if (system_prompt && strlen(system_prompt) > 0) {
        pos += snprintf(out + pos, out_sz - pos,
                        "System: ");
        const char *sp = system_prompt;
        while (*sp && pos < out_sz - 1) {
            if (*sp == '"' || *sp == '\\')
                out[pos++] = '\\';
            out[pos++] = *sp++;
        }
        pos += snprintf(out + pos, out_sz - pos, "\n\n");
    }

    if (messages_json && strlen(messages_json) > 2) {
        const char *p = messages_json;
        while ((p = strstr(p, "\"role\":\"")) != NULL) {
            p += strlen("\"role\":\"");
            char role[16] = "";
            size_t ri = 0;
            while (*p && *p != '"' && ri < sizeof(role) - 1)
                role[ri++] = *p++;
            role[ri] = '\0';

            /* Look for content field after this role */
            const char *c = strstr(p, "\"content\":");
            if (!c) break;
            c += strlen("\"content\":");

            /* Skip whitespace */
            while (*c == ' ' || *c == '\t') c++;

            /* If content is null, skip this message (tool_calls entry) */
            if (strncmp(c, "null", 4) == 0) {
                p = c + 4;
                continue;
            }

            /* Must be a quoted string */
            if (*c != '"') {
                p = c;
                continue;
            }
            c++; /* skip opening quote */

            pos += snprintf(out + pos, out_sz - pos, "%s: ", role);
            while (*c && pos < out_sz - 1) {
                if (c[0] == '\\' && c[1] == '"') {
                    out[pos++] = '"'; c += 2;
                } else if (c[0] == '\\' && c[1] == 'n') {
                    out[pos++] = '\n'; c += 2;
                } else if (c[0] == '\\' && c[1] == '\\') {
                    out[pos++] = '\\'; c += 2;
                } else if (c[0] == '"') {
                    break;
                } else {
                    out[pos++] = *c++;
                }
            }
            pos += snprintf(out + pos, out_sz - pos, "\n");
        }
    }

    out[pos] = '\0';
    return (int)pos;
}

static esp_err_t ollama_complete(
    const char *system_prompt,
    const char *messages_json,
    const char *tools_json,
    char       *response_buf,
    size_t      response_sz)
{
    (void)tools_json; /* Ollama /api/generate does not support tools in this mode */

    /* Build prompt */
    char *prompt = malloc(LLM_REQUEST_BUF_SIZE);
    if (!prompt) return ESP_ERR_NO_MEM;

    int plen = build_prompt(system_prompt, messages_json, prompt, LLM_REQUEST_BUF_SIZE);
    if (plen <= 0) {
        free(prompt);
        return ESP_ERR_INVALID_ARG;
    }

    /* Build JSON body */
    char *body = malloc(LLM_REQUEST_BUF_SIZE);
    if (!body) { free(prompt); return ESP_ERR_NO_MEM; }

    int len = snprintf(body, LLM_REQUEST_BUF_SIZE,
        "{"
        "\"model\":\"%s\","
        "\"prompt\":\"",
        s_model);

    /* JSON-escape the prompt into the body */
    const char *pp = prompt;
    while (*pp && len < LLM_REQUEST_BUF_SIZE - 32) {
        unsigned char c = (unsigned char)*pp++;
        if      (c == '"')  { body[len++] = '\\'; body[len++] = '"';  }
        else if (c == '\\') { body[len++] = '\\'; body[len++] = '\\'; }
        else if (c == '\n') { body[len++] = '\\'; body[len++] = 'n';  }
        else if (c == '\r') { body[len++] = '\\'; body[len++] = 'r';  }
        else if (c == '\t') { body[len++] = '\\'; body[len++] = 't';  }
        else                { body[len++] = (char)c; }
    }
    free(prompt);

    len += snprintf(body + len, LLM_REQUEST_BUF_SIZE - len,
        "\",\"stream\":false}");

    ESP_LOGI(TAG, "Request body (%d bytes): %.300s%s", len, body,
             len > 300 ? "..." : "");

    if (!espclaw_tls_lock(pdMS_TO_TICKS(60000))) {
        ESP_LOGW(TAG, "TLS lock timeout");
        free(body);
        return ESP_ERR_TIMEOUT;
    }

    /* Allocate response buffer */
    char *resp = malloc(LLM_RESPONSE_BUF_SIZE);
    if (!resp) { espclaw_tls_unlock(); free(body); return ESP_ERR_NO_MEM; }
    resp[0] = '\0';

    http_ctx_t ctx = { .buf = resp, .buf_sz = LLM_RESPONSE_BUF_SIZE };

    esp_http_client_config_t cfg = {
        .url              = s_base_url,
        .method           = HTTP_METHOD_POST,
        .timeout_ms       = LLM_HTTP_TIMEOUT_MS,
        .crt_bundle_attach= esp_crt_bundle_attach,
        .event_handler    = http_event_handler,
        .user_data        = &ctx,
        .buffer_size      = 2048,
        .buffer_size_tx   = 2048,
    };

    esp_http_client_handle_t client = esp_http_client_init(&cfg);
    if (!client) { espclaw_tls_unlock(); free(body); free(resp); return ESP_FAIL; }

    esp_http_client_set_header(client, "content-type", "application/json; charset=utf-8");
    esp_http_client_set_post_field(client, body, len);

    esp_err_t err = esp_http_client_perform(client);
    int status    = esp_http_client_get_status_code(client);

    if (err != ESP_OK || status != 200) {
        ESP_LOGE(TAG, "HTTP %d: %s, status=%d", err, esp_err_to_name(err), status);
        ESP_LOGE(TAG, "Response: %.500s", resp);
    }

    esp_http_client_cleanup(client);
    espclaw_tls_unlock();
    free(body);

    if (err != ESP_OK || status != 200) {
        free(resp);
        return ESP_FAIL;
    }

    extract_response(resp, response_buf, response_sz);
    ESP_LOGI(TAG, "Got %d chars", (int)strlen(response_buf));

    free(resp);
    return ESP_OK;
}

const provider_ops_t ollama_native_provider = {
    .name     = "ollama",
    .init     = ollama_init,
    .complete = ollama_complete,
    .deinit   = NULL,
};
