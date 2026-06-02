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
	"fmt"
	"iter"
	"net/http"
	"strings"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// Config configures an OpenAI model.LLM.
type Config struct {
	// APIKey is the OpenAI API key. If empty, the OpenAI SDK will use
	// OPENAI_API_KEY from the environment.
	APIKey string
	// BaseURL overrides the OpenAI API base URL. It defaults to
	// https://api.openai.com/v1. If the supplied URL does not end in /v1, /v1 is
	// appended.
	BaseURL string
	// HTTPClient overrides the HTTP client used by the OpenAI SDK.
	HTTPClient *http.Client
	// Headers are added to every OpenAI request.
	Headers http.Header
	// MaxRetries configures SDK retries. If nil, the SDK default is used.
	MaxRetries *int
}

// Option configures an OpenAI model.
type Option func(*Config)

// WithAPIKey configures the OpenAI API key. If omitted, OPENAI_API_KEY is used
// by the OpenAI SDK.
func WithAPIKey(apiKey string) Option {
	return func(c *Config) { c.APIKey = apiKey }
}

// WithBaseURL configures a custom OpenAI-compatible base URL.
func WithBaseURL(baseURL string) Option {
	return func(c *Config) { c.BaseURL = baseURL }
}

// WithHTTPClient configures the HTTP client used by the OpenAI SDK.
func WithHTTPClient(client *http.Client) Option {
	return func(c *Config) { c.HTTPClient = client }
}

// WithHeaders configures request headers added to every OpenAI request.
func WithHeaders(headers http.Header) Option {
	return func(c *Config) {
		if headers == nil {
			return
		}
		if c.Headers == nil {
			c.Headers = make(http.Header)
		}
		for k, values := range headers {
			for _, v := range values {
				c.Headers.Add(k, v)
			}
		}
	}
}

// WithMaxRetries configures the maximum SDK retry count.
func WithMaxRetries(maxRetries int) Option {
	return func(c *Config) { c.MaxRetries = &maxRetries }
}

type openAIModel struct {
	client openaisdk.Client
	name   string
	cfg    Config
}

// NewModel returns [model.LLM], backed by the Responses API.
//
// It uses the provided context and configuration to initialize the underlying
// [openAIModel]. The modelName specifies which OpenAI model to target
// (e.g., "gpt-5.4-mini").
//
// An error is returned if the [openAIModel] fails to initialize.
// NewModel returns a [model.LLM] backed by the OpenAI Responses API.
func NewModel(_ context.Context, modelName string, opts ...Option) (model.LLM, error) {
	if strings.TrimSpace(modelName) == "" {
		return nil, ErrModelNameRequired
	}

	cfg := Config{BaseURL: defaultBaseURL}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}

	requestOpts := []option.RequestOption{}
	if cfg.APIKey != "" {
		requestOpts = append(requestOpts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.BaseURL != "" {
		requestOpts = append(requestOpts, option.WithBaseURL(normalizeBaseURL(cfg.BaseURL)))
	}
	if cfg.HTTPClient != nil {
		requestOpts = append(requestOpts, option.WithHTTPClient(cfg.HTTPClient))
	}
	if cfg.MaxRetries != nil {
		requestOpts = append(requestOpts, option.WithMaxRetries(*cfg.MaxRetries))
	}
	for k, values := range cfg.Headers {
		for _, v := range values {
			requestOpts = append(requestOpts, option.WithHeaderAdd(k, v))
		}
	}

	return &openAIModel{
		client: openaisdk.NewClient(requestOpts...),
		name:   modelName,
		cfg:    cfg,
	}, nil
}

func (m *openAIModel) Name() string { return m.name }

// GenerateContent converts req to an OpenAI Responses API request and calls
// /v1/responses. When stream is true it uses Responses streaming.
func (m *openAIModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if req == nil {
		return singleErrorSequence(ErrRequestNil)
	}

	params, err := m.buildParams(req)
	if err != nil {
		return singleErrorSequence(err)
	}
	if stream {
		return m.generateStream(ctx, params)
	}
	return m.generate(ctx, params)
}

func (m *openAIModel) buildParams(req *model.LLMRequest) (responses.ResponseNewParams, error) {
	modelName := strings.TrimSpace(req.Model)
	if modelName == "" {
		modelName = m.name
	}

	input, instructions, err := contentsToResponsesInput(req.Contents, req.Config)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}

	params := responses.ResponseNewParams{
		Model: modelName,
		Input: input,
	}
	if instructions != "" {
		params.Instructions = param.NewOpt(instructions)
	}
	params.Store = param.NewOpt(false)
	params.ParallelToolCalls = param.NewOpt(true)
	params.Include = append(params.Include, responses.ResponseIncludableReasoningEncryptedContent)

	if err := applyGenerationConfig(&params, req.Config); err != nil {
		return responses.ResponseNewParams{}, err
	}
	return params, nil
}

func (m *openAIModel) generate(ctx context.Context, params responses.ResponseNewParams) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.client.Responses.New(ctx, params)
		if err != nil {
			yield(nil, fmt.Errorf("%w: %w", ErrResponsesAPIFailed, err))
			return
		}

		llmResp, convErr := llmResponseFromResponse(resp)
		if convErr != nil {
			yield(nil, convErr)
			return
		}
		yield(llmResp, nil)
	}
}

func (m *openAIModel) generateStream(ctx context.Context, params responses.ResponseNewParams) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.runStreaming(ctx, params, yield)
	}
}

func normalizeBaseURL(baseURL string) string {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if baseURL == "" {
		return defaultBaseURL
	}
	lower := strings.ToLower(baseURL)
	if strings.HasSuffix(lower, "/v1") || strings.Contains(lower, "/v1/") {
		return baseURL
	}
	return baseURL + "/v1"
}

func singleErrorSequence(err error) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(nil, err)
	}
}

func applyGenerationConfig(params *responses.ResponseNewParams, cfg *genai.GenerateContentConfig) error {
	if cfg == nil {
		return nil
	}
	if cfg.TopK != nil {
		return ErrTopKNotSupported
	}
	if len(cfg.StopSequences) > 0 {
		return ErrStopSequencesNotSupported
	}
	if cfg.CandidateCount > 1 {
		return ErrMultipleCandidatesNotSupported
	}
	if cfg.FrequencyPenalty != nil || cfg.PresencePenalty != nil {
		return ErrPenaltiesNotSupported
	}
	if len(cfg.Labels) > 0 {
		return ErrLabelsNotSupported
	}
	if len(cfg.SafetySettings) > 0 {
		return ErrSafetySettingsNotSupported
	}
	if cfg.Temperature != nil {
		params.Temperature = param.NewOpt(float64(*cfg.Temperature))
	}
	if cfg.TopP != nil {
		params.TopP = param.NewOpt(float64(*cfg.TopP))
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(cfg.MaxOutputTokens))
	}
	if cfg.Logprobs != nil {
		params.TopLogprobs = param.NewOpt(int64(*cfg.Logprobs))
	}
	if cfg.ResponseLogprobs {
		params.Include = append(params.Include, responses.ResponseIncludableMessageOutputTextLogprobs)
	}
	if cfg.Tools != nil {
		tools, err := genaiToolsToResponses(cfg.Tools)
		if err != nil {
			return err
		}
		params.Tools = tools
	}
	if cfg.ToolConfig != nil && cfg.ToolConfig.FunctionCallingConfig != nil {
		toolChoice, err := toolChoiceFromGenai(cfg.ToolConfig.FunctionCallingConfig)
		if err != nil {
			return err
		}
		if toolChoice != nil {
			params.ToolChoice = *toolChoice
		}
	}
	if cfg.ThinkingConfig != nil {
		if effort := reasoningEffortFromThinkingConfig(cfg.ThinkingConfig); effort != "" {
			params.Reasoning = shared.ReasoningParam{Effort: effort}
		}
	}
	if len(cfg.Labels) > 0 {
		params.Metadata = make(shared.Metadata, len(cfg.Labels))
		for k, v := range cfg.Labels {
			params.Metadata[k] = v
		}
	}
	if cfg.ResponseMIMEType != "" || cfg.ResponseSchema != nil || cfg.ResponseJsonSchema != nil {
		textConfig, err := responseTextConfig(cfg)
		if err != nil {
			return err
		}
		if textConfig != nil {
			params.Text = *textConfig
		}
	}
	return nil
}

func toolChoiceFromGenai(cfg *genai.FunctionCallingConfig) (*responses.ResponseNewParamsToolChoiceUnion, error) {
	if cfg == nil {
		return nil, nil
	}
	if len(cfg.AllowedFunctionNames) > 0 {
		allowed := make([]map[string]any, 0, len(cfg.AllowedFunctionNames))
		for _, name := range cfg.AllowedFunctionNames {
			if strings.TrimSpace(name) == "" {
				continue
			}
			allowed = append(allowed, map[string]any{mapKeyType: Function, mapKeyName: name})
		}
		if len(allowed) == 0 {
			return nil, nil
		}
		mode := responses.ToolChoiceAllowedModeAuto
		if cfg.Mode == genai.FunctionCallingConfigModeAny {
			mode = responses.ToolChoiceAllowedModeRequired
		}
		return &responses.ResponseNewParamsToolChoiceUnion{OfAllowedTools: &responses.ToolChoiceAllowedParam{Mode: mode, Tools: allowed}}, nil
	}
	switch cfg.Mode {
	case genai.FunctionCallingConfigModeNone:
		return &responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone)}, nil
	case genai.FunctionCallingConfigModeAny:
		return &responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired)}, nil
	case genai.FunctionCallingConfigModeAuto, genai.FunctionCallingConfigModeValidated:
		return &responses.ResponseNewParamsToolChoiceUnion{OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsAuto)}, nil
	default:
		return nil, fmt.Errorf("%w %q", ErrUnsupportedToolCallingMode, cfg.Mode)
	}
}

func reasoningEffortFromThinkingConfig(cfg *genai.ThinkingConfig) shared.ReasoningEffort {
	if cfg == nil {
		return ""
	}
	if cfg.ThinkingLevel != genai.ThinkingLevelUnspecified {
		switch cfg.ThinkingLevel {
		case genai.ThinkingLevelMinimal:
			return shared.ReasoningEffortMinimal
		case genai.ThinkingLevelLow:
			return shared.ReasoningEffortLow
		case genai.ThinkingLevelMedium:
			return shared.ReasoningEffortMedium
		case genai.ThinkingLevelHigh:
			return shared.ReasoningEffortHigh
		}
	}
	if cfg.ThinkingBudget != nil {
		budget := *cfg.ThinkingBudget
		switch {
		case budget >= 8000:
			return shared.ReasoningEffortHigh
		case budget >= 2000:
			return shared.ReasoningEffortMedium
		case budget > 0:
			return shared.ReasoningEffortLow
		case budget == 0:
			return shared.ReasoningEffortNone
		}
	}
	return ""
}

func responseTextConfig(cfg *genai.GenerateContentConfig) (*responses.ResponseTextConfigParam, error) {
	if cfg == nil {
		return nil, nil
	}
	if cfg.ResponseMIMEType == "" || cfg.ResponseMIMEType == textPlainMIMEType {
		if cfg.ResponseSchema == nil && cfg.ResponseJsonSchema == nil {
			return nil, nil
		}
	}
	if cfg.ResponseMIMEType != "" && cfg.ResponseMIMEType != applicationJSONMIMEType && cfg.ResponseMIMEType != textPlainMIMEType {
		return nil, fmt.Errorf("%w %q", ErrUnsupportedResponseMIMEType, cfg.ResponseMIMEType)
	}
	if cfg.ResponseMIMEType == applicationJSONMIMEType && cfg.ResponseSchema == nil && cfg.ResponseJsonSchema == nil {
		return nil, ErrJSONResponseWithoutSchema
	}
	if cfg.ResponseJsonSchema != nil {
		schema := schemaMap(cfg.ResponseJsonSchema)
		return &responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
					Name:   resolvedResponseSchemaName(cfg),
					Schema: schema,
					Strict: param.NewOpt(true),
				},
			},
		}, nil
	}
	if cfg.ResponseSchema != nil {
		schema := schemaMap(cfg.ResponseSchema)
		return &responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
					Name:   resolvedResponseSchemaName(cfg),
					Schema: schema,
					Strict: param.NewOpt(true),
				},
			},
		}, nil
	}
	return &responses.ResponseTextConfigParam{
		Format: responses.ResponseFormatTextConfigUnionParam{
			OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
		},
	}, nil
}

func resolvedResponseSchemaName(cfg *genai.GenerateContentConfig) string {
	if cfg != nil && cfg.ResponseSchema != nil && strings.TrimSpace(cfg.ResponseSchema.Title) != "" {
		return cfg.ResponseSchema.Title
	}
	return defaultResponseSchemaName
}
