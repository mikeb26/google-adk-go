// Copyright 2026 Mike Brown
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestNewModel_GenerateContent(t *testing.T) {
	server := newLocalhostServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.URL.Path, "/v1/responses"; got != want {
			t.Fatalf("unexpected path: got %s want %s", got, want)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":"resp_123","model":"test-model","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
	}))
	defer server.Close()

	llm, err := NewModel(context.Background(), "gpt-4o-mini",
		WithAPIKey("test"),
		WithHTTPClient(server.Client()),
		WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	var gotText string
	for resp, err := range llm.GenerateContent(context.Background(), &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
	}, false) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		if resp != nil && resp.Content != nil {
			for _, part := range resp.Content.Parts {
				gotText += part.Text
			}
		}
	}

	if got, want := gotText, "hello"; got != want {
		t.Fatalf("text mismatch: got %q want %q", got, want)
	}
}

func TestNewModel_GenerateContent_Stream(t *testing.T) {
	server := newLocalhostServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.URL.Path, "/v1/responses"; got != want {
			t.Fatalf("unexpected path: got %s want %s", got, want)
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("failed to unmarshal request body: %v", err)
		}
		if stream, _ := payload["stream"].(bool); !stream {
			t.Fatalf("expected request body to set stream=true, got: %s", string(body))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"content_index\":0,\"delta\":\"hel\",\"item_id\":\"item_123\",\"logprobs\":[],\"output_index\":0,\"sequence_number\":1,\"type\":\"response.output_text.delta\"}\n\n"+
			"data: {\"content_index\":0,\"delta\":\"lo\",\"item_id\":\"item_123\",\"logprobs\":[],\"output_index\":0,\"sequence_number\":2,\"type\":\"response.output_text.delta\"}\n\n"+
			"data: {\"response\":{\"id\":\"resp_123\",\"model\":\"test-model\",\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}},\"sequence_number\":3,\"type\":\"response.completed\"}\n\n")
	}))
	defer server.Close()

	llm, err := NewModel(context.Background(), "gpt-4o-mini",
		WithAPIKey("test"),
		WithHTTPClient(server.Client()),
		WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("NewModel() error = %v", err)
	}

	var partialText, finalText string
	var sawPartial, sawFinal bool
	for resp, err := range llm.GenerateContent(context.Background(), &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
	}, true) {
		if err != nil {
			t.Fatalf("GenerateContent() error = %v", err)
		}
		if resp == nil || resp.Content == nil || len(resp.Content.Parts) == 0 {
			t.Fatalf("GenerateContent() yielded empty response: %+v", resp)
		}

		text := resp.Content.Parts[0].Text
		if resp.Partial {
			sawPartial = true
			partialText += text
		} else {
			sawFinal = true
			finalText += text
		}
	}

	if !sawPartial {
		t.Fatal("expected at least one partial streaming response")
	}
	if !sawFinal {
		t.Fatal("expected a final streaming response")
	}
	if got, want := partialText, "hello"; got != want {
		t.Fatalf("partial text mismatch: got %q want %q", got, want)
	}
	if got, want := finalText, "hello"; got != want {
		t.Fatalf("final text mismatch: got %q want %q", got, want)
	}
}

func TestApplyGenerationConfig_Validation(t *testing.T) {
	base := &responses.ResponseNewParams{}
	tests := []struct {
		name string
		cfg  *genai.GenerateContentConfig
		want error
	}{
		{name: "topk", cfg: &genai.GenerateContentConfig{TopK: float32Ptr(1)}, want: ErrTopKNotSupported},
		{name: "stop sequences", cfg: &genai.GenerateContentConfig{StopSequences: []string{"done"}}, want: ErrStopSequencesNotSupported},
		{name: "candidate count", cfg: &genai.GenerateContentConfig{CandidateCount: 2}, want: ErrMultipleCandidatesNotSupported},
		{name: "penalties", cfg: &genai.GenerateContentConfig{FrequencyPenalty: float32Ptr(1)}, want: ErrPenaltiesNotSupported},
		{name: "labels", cfg: &genai.GenerateContentConfig{Labels: map[string]string{"a": "b"}}, want: ErrLabelsNotSupported},
		{name: "safety settings", cfg: &genai.GenerateContentConfig{SafetySettings: []*genai.SafetySetting{{Category: genai.HarmCategoryHateSpeech}}}, want: ErrSafetySettingsNotSupported},
		{name: "unsupported mime", cfg: &genai.GenerateContentConfig{ResponseMIMEType: "application/xml"}, want: ErrUnsupportedResponseMIMEType},
		{name: "json without schema", cfg: &genai.GenerateContentConfig{ResponseMIMEType: "application/json"}, want: ErrJSONResponseWithoutSchema},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := *base
			err := applyGenerationConfig(&params, tt.cfg)
			if !errors.Is(err, tt.want) {
				t.Fatalf("applyGenerationConfig() error = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestApplyGenerationConfig_Combinations(t *testing.T) {
	tests := []struct {
		name                string
		cfg                 *genai.GenerateContentConfig
		wantTemperature     *float64
		wantTopP            *float64
		wantMaxOutput       *int64
		wantTopLogprobs     *int64
		wantIncludeLogprobs bool
		wantReasoning       shared.ReasoningEffort
	}{
		{
			name:                "sampling knobs and logprobs",
			cfg:                 &genai.GenerateContentConfig{Temperature: float32Ptr(0.7), TopP: float32Ptr(0.9), MaxOutputTokens: 128, Logprobs: int32Ptr(5), ResponseLogprobs: true},
			wantTemperature:     float64Ptr(0.7),
			wantTopP:            float64Ptr(0.9),
			wantMaxOutput:       int64Ptr(128),
			wantTopLogprobs:     int64Ptr(5),
			wantIncludeLogprobs: true,
		},
		{
			name:            "reasoning level minimal wins over budget",
			cfg:             &genai.GenerateContentConfig{Temperature: float32Ptr(0.2), ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMinimal, ThinkingBudget: int32Ptr(8000)}},
			wantTemperature: float64Ptr(0.2),
			wantReasoning:   shared.ReasoningEffortMinimal,
		},
		{
			name:          "reasoning level high",
			cfg:           &genai.GenerateContentConfig{TopP: float32Ptr(0.8), ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh}},
			wantTopP:      float64Ptr(0.8),
			wantReasoning: shared.ReasoningEffortHigh,
		},
		{
			name:          "reasoning budget low",
			cfg:           &genai.GenerateContentConfig{MaxOutputTokens: 256, ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: int32Ptr(1)}},
			wantMaxOutput: int64Ptr(256),
			wantReasoning: shared.ReasoningEffortLow,
		},
		{
			name:            "reasoning budget medium",
			cfg:             &genai.GenerateContentConfig{Logprobs: int32Ptr(3), ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: int32Ptr(2500)}},
			wantTopLogprobs: int64Ptr(3),
			wantReasoning:   shared.ReasoningEffortMedium,
		},
		{
			name:                "reasoning budget high and zero means none",
			cfg:                 &genai.GenerateContentConfig{ResponseLogprobs: true, ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: int32Ptr(8000)}},
			wantIncludeLogprobs: true,
			wantReasoning:       shared.ReasoningEffortHigh,
		},
		{
			name:          "reasoning budget zero maps to none",
			cfg:           &genai.GenerateContentConfig{ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: int32Ptr(0)}},
			wantReasoning: shared.ReasoningEffortNone,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := &responses.ResponseNewParams{}
			if err := applyGenerationConfig(params, tt.cfg); err != nil {
				t.Fatalf("applyGenerationConfig() error = %v", err)
			}

			assertFloatOpt(t, "temperature", params.Temperature, tt.wantTemperature)
			assertFloatOpt(t, "top_p", params.TopP, tt.wantTopP)
			assertInt64Opt(t, "max_output_tokens", params.MaxOutputTokens, tt.wantMaxOutput)
			assertInt64Opt(t, "top_logprobs", params.TopLogprobs, tt.wantTopLogprobs)

			if tt.wantIncludeLogprobs {
				if len(params.Include) != 1 || params.Include[0] != responses.ResponseIncludableMessageOutputTextLogprobs {
					t.Fatalf("unexpected include values: %+v", params.Include)
				}
			} else if len(params.Include) != 0 {
				t.Fatalf("unexpected include values: %+v", params.Include)
			}

			if got, want := params.Reasoning.Effort, tt.wantReasoning; got != want {
				t.Fatalf("reasoning effort mismatch: got %q want %q", got, want)
			}
		})
	}
}

func TestGenaiToolsToResponses_ValidationAndStrictness(t *testing.T) {
	t.Run("rejects non-function tools", func(t *testing.T) {
		_, err := genaiToolsToResponses([]*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}})
		if !errors.Is(err, ErrNonFunctionToolUnsupported) {
			t.Fatalf("genaiToolsToResponses() error = %v, want %v", err, ErrNonFunctionToolUnsupported)
		}
	})

	t.Run("accepts function tools with strict schemas", func(t *testing.T) {
		tools, err := genaiToolsToResponses([]*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{{
			Name:        "lookup",
			Description: "look something up",
			Parameters: &genai.Schema{
				Type:       genai.TypeObject,
				Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
			},
		}}}})
		if err != nil {
			t.Fatalf("genaiToolsToResponses() error = %v", err)
		}
		if len(tools) != 1 || tools[0].OfFunction == nil {
			t.Fatalf("unexpected tools: %+v", tools)
		}
		if !tools[0].OfFunction.Strict.Valid() || !tools[0].OfFunction.Strict.Value {
			t.Fatalf("expected strict function tool, got %+v", tools[0].OfFunction)
		}
	})
}

func TestResponseTextConfig_SchemaModes(t *testing.T) {
	t.Run("response schema", func(t *testing.T) {
		cfg := &genai.GenerateContentConfig{
			ResponseMIMEType: "application/json",
			ResponseSchema: &genai.Schema{
				Title: "WeatherResponse",
				Type:  genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"answer": {Type: genai.TypeString},
				},
			},
		}
		textCfg, err := responseTextConfig(cfg)
		if err != nil {
			t.Fatalf("responseTextConfig() error = %v", err)
		}
		if textCfg == nil || textCfg.Format.OfJSONSchema == nil {
			t.Fatalf("expected json schema config, got %+v", textCfg)
		}
		if got, want := textCfg.Format.OfJSONSchema.Name, "WeatherResponse"; got != want {
			t.Fatalf("schema name mismatch: got %q want %q", got, want)
		}
		if !textCfg.Format.OfJSONSchema.Strict.Valid() || !textCfg.Format.OfJSONSchema.Strict.Value {
			t.Fatalf("expected strict schema, got %+v", textCfg.Format.OfJSONSchema)
		}
	})

	t.Run("response json schema", func(t *testing.T) {
		cfg := &genai.GenerateContentConfig{
			ResponseMIMEType: "application/json",
			ResponseJsonSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"answer": map[string]any{"type": "string"}},
			},
		}
		textCfg, err := responseTextConfig(cfg)
		if err != nil {
			t.Fatalf("responseTextConfig() error = %v", err)
		}
		if textCfg == nil || textCfg.Format.OfJSONSchema == nil {
			t.Fatalf("expected json schema config, got %+v", textCfg)
		}
		if textCfg.Format.OfJSONSchema.Schema["type"] != "object" {
			t.Fatalf("unexpected schema: %+v", textCfg.Format.OfJSONSchema.Schema)
		}
	})
}

func newLocalhostServer(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()
	server := httptest.NewUnstartedServer(handler)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen on IPv4 loopback: %v", err)
	}
	server.Listener = ln
	server.Start()
	return server
}

func float32Ptr(v float32) *float32 { return &v }

func float64Ptr(v float64) *float64 { return &v }

func int32Ptr(v int32) *int32 { return &v }

func int64Ptr(v int64) *int64 { return &v }

func assertFloatOpt(t *testing.T, name string, got param.Opt[float64], want *float64) {
	t.Helper()
	switch {
	case want == nil:
		if got.Valid() {
			t.Fatalf("%s: got %v, want unset", name, got.Value)
		}
	case !got.Valid():
		t.Fatalf("%s: got unset, want %v", name, *want)
	case !float64ApproxEqual(got.Value, *want):
		t.Fatalf("%s mismatch: got %v want %v", name, got.Value, *want)
	}
}

func assertInt64Opt(t *testing.T, name string, got param.Opt[int64], want *int64) {
	t.Helper()
	switch {
	case want == nil:
		if got.Valid() {
			t.Fatalf("%s: got %v, want unset", name, got.Value)
		}
	case !got.Valid():
		t.Fatalf("%s: got unset, want %v", name, *want)
	case got.Value != *want:
		t.Fatalf("%s mismatch: got %v want %v", name, got.Value, *want)
	}
}

func float64ApproxEqual(a, b float64) bool {
	const eps = 1e-6
	if a > b {
		return a-b <= eps
	}
	return b-a <= eps
}
