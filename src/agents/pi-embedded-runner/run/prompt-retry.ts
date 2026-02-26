import type { OutboundRetryConfig } from "../../../config/types.base.js";
import { retryAsync } from "../../../infra/retry.js";
import { isRateLimitErrorMessage } from "../../pi-embedded-helpers/errors.js";
import { log } from "../logger.js";

/**
 * Extract HTTP status code from error object.
 * Supports various error formats from different LLM providers.
 */
function extractStatusCode(err: unknown): number | undefined {
  if (typeof err !== "object" || err === null) {
    return undefined;
  }

  const errObj = err as Record<string, unknown>;

  // Direct status property
  if (typeof errObj.status === "number") {
    return errObj.status;
  }
  if (typeof errObj.status === "string") {
    const parsed = Number(errObj.status);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }

  // Nested in error object
  if (errObj.error && typeof errObj.error === "object") {
    const nested = errObj.error as Record<string, unknown>;
    if (typeof nested.status === "number") {
      return nested.status;
    }
    if (typeof nested.status === "string") {
      const parsed = Number(nested.status);
      if (!Number.isNaN(parsed)) {
        return parsed;
      }
    }
  }

  // HTTP style code in message (e.g., "HTTP 429")
  const msg = extractErrorMessage(err);
  const match = msg.match(/HTTP[^\d]*(\d{3})/i);
  if (match) {
    return Number(match[1]);
  }

  return undefined;
}

/**
 * Check if HTTP status code indicates a retryable error.
 * These codes represent temporary failures that may resolve on retry.
 */
function isRetryableStatusCode(code: number): boolean {
  // 429: Rate limit exceeded
  // 500: Internal server error
  // 502: Bad gateway (temporary overload)
  // 503: Service unavailable (temporary overload)
  // 504: Gateway timeout (temporary)
  const retryableCodes = [429, 500, 502, 503, 504];
  return retryableCodes.includes(code);
}

/**
 * Extract error message from various error formats across different LLM providers.
 * Handles:
 * - String errors: "TPM limit exceeded"
 * - Error objects: new Error("message") or { message: "message" }
 * - API response objects: { type: "error", error: { type: "rate_limit_error", message: "..." } }
 * - JSON stringified errors: '{"type":"error",...}'
 */
function extractErrorMessage(err: unknown): string {
  if (err === null || err === undefined) {
    return "";
  }
  // Handle string errors
  if (typeof err === "string") {
    return err;
  }
  // Handle Error objects
  if (err instanceof Error && err.message) {
    return err.message;
  }
  // Handle plain objects with error properties
  if (typeof err === "object") {
    const errObj = err as Record<string, unknown>;
    // Try nested error structure (common in Anthropic/OpenAI SDK)
    if (errObj.error && typeof errObj.error === "object") {
      const nested = errObj.error as Record<string, unknown>;
      // eslint-disable-next-line @typescript-eslint/no-base-to-string
      for (const key of ["message", "error", "type", "code"]) {
        const value = nested[key] ?? errObj[key];
        if (typeof value === "string" && value) {
          return value;
        }
      }
      // Fallback to JSON stringification
      return JSON.stringify(errObj.error);
    }
    // Direct properties
    // eslint-disable-next-line @typescript-eslint/no-base-to-string
    for (const key of ["message", "error", "code", "reason", "type"]) {
      const value = errObj[key];
      if (typeof value === "string" && value) {
        return value;
      }
    }
    // Fallback to JSON stringification
    return JSON.stringify(errObj);
  }
  // eslint-disable-next-line @typescript-eslint/no-base-to-string
  return String(err);
}

/**
 * Check if error is retryable (rate limit, overload, or temporary failure).
 * Supports all major LLM providers and handles both SDK errors and formatted
 * user-facing messages (e.g., from pi-ai SDK after retries exhausted).
 */
function isRetryableError(err: unknown): boolean {
  // Check HTTP status code first (most reliable for retries)
  const statusCode = extractStatusCode(err);
  if (statusCode !== undefined && isRetryableStatusCode(statusCode)) {
    return true;
  }

  const msg = extractErrorMessage(err).toLowerCase();

  // Reuse the core rate limit pattern detection from errors.ts
  if (isRateLimitErrorMessage(msg)) {
    return true;
  }

  // SDK-specific error types not covered by isRateLimitErrorMessage
  // These are explicit error types from various LLM SDKs
  const sdkErrorTypes = [
    "rate_limit_error",
    "rate_limit_exceeded",
    "overloaded_error",
    "throttling_exception",
    "resource_exhausted",
    "resource_has_been_exhausted",
    "tokens_per_minute",
  ];

  for (const type of sdkErrorTypes) {
    if (msg.includes(type)) {
      return true;
    }
  }

  // Chinese error patterns for LLM providers
  const chineseErrorPatterns = ["请求额度超限", "请求频率超限", "限流", "速率限制"];

  for (const pattern of chineseErrorPatterns) {
    if (msg.includes(pattern)) {
      return true;
    }
  }

  // Check for formatted error messages (from pi-ai SDK or user-facing errors)
  // These are often what remains after SDK retries have been exhausted
  const formattedErrorPatterns = [
    /api[_\s]?rate[_\s]?limit/i,
    /api rate limit reached/i,
    /too[_\s]?many[_\s]?requests?/i,
    /tpm[_\s]?limit/i,
    /tokens per minute/i,
    /rate[_\s]?limit[_\s]?(?:exceeded|error)?/i,
    /overloaded/i,
    /service[_\s]?(?:unavailable|temporarily[_\s]?overloaded)/i,
    /请/i, // Chinese "please" (part of "请稍后重试" type messages)
    /重试/i, // Chinese "retry"
  ];

  if (formattedErrorPatterns.some((pattern) => pattern.test(msg))) {
    return true;
  }

  // HTTP 502/503 errors often indicate temporary service overload
  const httpErrorPatterns = [
    /\b502\b.*\bBad\s*Gateway\b/i,
    /\b503\b.*\bService\s*(?:Unavailable|Temporarily\s*Overloaded)/i,
  ];

  return httpErrorPatterns.some((pattern) => pattern.test(msg));
}

/**
 * Extract retry_after value from error for appropriate backoff.
 */
function getRetryAfterMs(err: unknown): number | undefined {
  const msg = extractErrorMessage(err);

  // Explicit retry_after field
  if (typeof err === "object" && err !== null) {
    const errObj = err as Record<string, unknown>;
    // Direct retry_after property
    if (typeof errObj.retry_after === "number") {
      return errObj.retry_after * 1000;
    }
    if (typeof errObj.retry_after === "string") {
      const parsed = Number(errObj.retry_after);
      if (!Number.isNaN(parsed)) {
        return parsed * 1000;
      }
    }
    // Nested in error object
    if (errObj.error && typeof errObj.error === "object") {
      const nested = errObj.error as Record<string, unknown>;
      if (typeof nested.retry_after === "number") {
        return nested.retry_after * 1000;
      }
      if (typeof nested.retry_after === "string") {
        const parsed = Number(nested.retry_after);
        if (!Number.isNaN(parsed)) {
          return parsed * 1000;
        }
      }
    }
  }

  // Match "retry_after: N" or "retry after N" in message
  const match = msg.match(/retry_after[:\s]*(\d+)/i) ?? msg.match(/retry\s*after[:\s]*(\d+)/i);
  if (match) {
    return Number(match[1]) * 1000;
  }

  return undefined;
}

export function getRetryConfig(
  provider: string,
  config?: { models?: { providers?: Record<string, { retry?: OutboundRetryConfig }> } },
): OutboundRetryConfig | undefined {
  return config?.models?.providers?.[provider]?.retry;
}

export async function runWithPromptRetry<T>(
  fn: () => Promise<T>,
  _provider: string,
  _modelId: string,
  retryConfig?: OutboundRetryConfig,
  signal?: AbortSignal,
): Promise<T> {
  // If no retry config is provided, run the function without retry logic (default: disabled)
  if (!retryConfig) {
    return fn();
  }

  const attempts = retryConfig.attempts ?? 3;
  const minDelayMs = retryConfig.minDelayMs ?? 1000;
  const maxDelayMs = retryConfig.maxDelayMs ?? 60000;
  const jitter = retryConfig.jitter ?? 0.2;

  return retryAsync(fn, {
    attempts,
    minDelayMs,
    maxDelayMs,
    jitter,
    shouldRetry: isRetryableError,
    retryAfterMs: getRetryAfterMs,
    onRetry: (info) => {
      log.warn(
        `[prompt-retry] retry attempt=${info.attempt}/${info.maxAttempts} delay=${info.delayMs}ms`,
      );
    },
    signal,
  });
}

export type { OutboundRetryConfig };
