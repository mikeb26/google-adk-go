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
	"encoding/json"
	"strings"

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

func llmResponseFromResponse(resp *responses.Response) (*model.LLMResponse, error) {
	if resp == nil {
		return nil, ErrEmptyResponse
	}
	if len(resp.Output) == 0 {
		if resp.Error.Code != "" || resp.Error.Message != "" {
			llmResp := &model.LLMResponse{ErrorCode: string(resp.Error.Code), ErrorMessage: resp.Error.Message}
			attachResponseMetadata(llmResp, resp)
			return llmResp, nil
		}
		return nil, ErrNoOutputItems
	}

	parts, finishReason, hasContent := partsFromResponseOutput(resp.Output)
	if !hasContent {
		return nil, ErrNoTextOrToolContent
	}
	llmResp := &model.LLMResponse{
		Partial:       false,
		TurnComplete:  true,
		FinishReason:  finishReasonFromResponse(resp, finishReason),
		UsageMetadata: usageMetadata(resp.Usage),
		ModelVersion:  string(resp.Model),
		Content:       &genai.Content{Role: string(genai.RoleModel), Parts: parts},
	}
	attachResponseMetadata(llmResp, resp)
	if resp.Error.Code != "" || resp.Error.Message != "" {
		llmResp.ErrorCode = string(resp.Error.Code)
		llmResp.ErrorMessage = resp.Error.Message
	}
	return llmResp, nil
}

func partsFromResponseOutput(items []responses.ResponseOutputItemUnion) ([]*genai.Part, string, bool) {
	var parts []*genai.Part
	var finishReason string
	hasContent := false

	for _, item := range items {
		switch variant := item.AsAny().(type) {
		case responses.ResponseOutputMessage:
			for _, content := range variant.Content {
				switch string(content.Type) {
				case OutputText:
					text := content.AsOutputText()
					if text.Text != "" {
						parts = append(parts, &genai.Part{Text: text.Text})
						hasContent = true
					}
				case Refusal:
					refusal := content.AsRefusal()
					if refusal.Refusal != "" {
						parts = append(parts, &genai.Part{Text: refusal.Refusal})
						hasContent = true
					}
				}
			}
		case responses.ResponseFunctionToolCall:
			parts = append(parts, functionCallPart(variant.Name, variant.CallID, variant.Arguments))
			hasContent = true
		case responses.ResponseFunctionToolCallOutputItem:
			parts = append(parts, functionResponsePart(variant.CallID, variant.Output))
			hasContent = true
		case responses.ResponseReasoningItem:
			parts = append(parts, reasoningPartFromResponseItem(variant))
			hasContent = true
		}
	}

	return parts, finishReason, hasContent
}

func functionCallPart(name, id, arguments string) *genai.Part {
	args := map[string]any{}
	if arguments != "" {
		_ = json.Unmarshal([]byte(arguments), &args)
	}
	part := genai.NewPartFromFunctionCall(name, args)
	if part.FunctionCall != nil {
		part.FunctionCall.ID = id
	}
	return part
}

func functionResponsePart(callID string, output any) *genai.Part {
	resp := map[string]any{}
	if output != nil {
		if m, ok := output.(map[string]any); ok {
			resp = m
		} else if s, ok := output.(string); ok {
			resp["output"] = s
		} else {
			resp["output"] = output
		}
	}
	part := genai.NewPartFromFunctionResponse(functionResponseToolName, resp)
	if part.FunctionResponse != nil {
		part.FunctionResponse.ID = callID
	}
	return part
}

func usageMetadata(usage responses.ResponseUsage) *genai.GenerateContentResponseUsageMetadata {
	if usage.InputTokens == 0 && usage.OutputTokens == 0 && usage.TotalTokens == 0 {
		return nil
	}
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(usage.InputTokens),
		CandidatesTokenCount: int32(usage.OutputTokens),
		TotalTokenCount:      int32(usage.TotalTokens),
	}
}

func attachResponseMetadata(llmResp *model.LLMResponse, resp *responses.Response) {
	if llmResp == nil || resp == nil {
		return
	}
	if llmResp.CustomMetadata == nil {
		llmResp.CustomMetadata = make(map[string]any)
	}
	if resp.ID != "" {
		llmResp.CustomMetadata[customMetadataResponseID] = resp.ID
	}
	if resp.Model != "" {
		llmResp.CustomMetadata[customMetadataModel] = string(resp.Model)
	}
	if resp.Status != "" {
		llmResp.CustomMetadata[customMetadataStatus] = string(resp.Status)
	}
}

func finishReasonFromResponse(resp *responses.Response, fallback string) genai.FinishReason {
	if resp == nil {
		return finishReasonFromString(fallback)
	}
	if resp.IncompleteDetails.Reason != "" {
		return finishReasonFromString(resp.IncompleteDetails.Reason)
	}
	if resp.Status != "" {
		return finishReasonFromString(string(resp.Status))
	}
	return finishReasonFromString(fallback)
}

func finishReasonFromString(reason string) genai.FinishReason {
	switch strings.ToLower(strings.TrimSpace(reason)) {
	case finishReasonMaxOutputTokens, finishReasonMaxTokens, finishReasonLength:
		return genai.FinishReasonMaxTokens
	case finishReasonContentFilter:
		return genai.FinishReasonSafety
	case finishReasonFailed, Error:
		return genai.FinishReasonOther
	default:
		return genai.FinishReasonStop
	}
}
