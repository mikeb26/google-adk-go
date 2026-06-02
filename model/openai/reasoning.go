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

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

type openAIReasoningContinuation struct {
	ID               string   `json:"id,omitempty"`
	Summary          []string `json:"summary,omitempty"`
	EncryptedContent string   `json:"encrypted_content,omitempty"`
}

func reasoningPartFromResponseItem(item responses.ResponseReasoningItem) *genai.Part {
	part := &genai.Part{Thought: true}
	if text := reasoningSummaryText(item.Summary); text != "" {
		part.Text = text
	}
	if meta := reasoningContinuationFromResponseItem(item); meta != nil {
		part.PartMetadata = map[string]any{customMetadataOpenAIReasoningContinuation: *meta}
	}
	return part
}

func reasoningContinuationFromResponseItem(item responses.ResponseReasoningItem) *openAIReasoningContinuation {
	if strings.TrimSpace(item.ID) == "" {
		return nil
	}
	return &openAIReasoningContinuation{
		ID:               strings.TrimSpace(item.ID),
		Summary:          reasoningSummaryTexts(item.Summary),
		EncryptedContent: item.EncryptedContent,
	}
}

func reasoningInputItemFromPart(part *genai.Part) (responses.ResponseInputItemUnionParam, bool) {
	if part == nil || part.PartMetadata == nil {
		return responses.ResponseInputItemUnionParam{}, false
	}
	raw, ok := part.PartMetadata[customMetadataOpenAIReasoningContinuation]
	if !ok {
		return responses.ResponseInputItemUnionParam{}, false
	}
	meta, ok := decodeReasoningContinuation(raw)
	if !ok {
		return responses.ResponseInputItemUnionParam{}, false
	}
	if strings.TrimSpace(meta.ID) == "" {
		return responses.ResponseInputItemUnionParam{}, false
	}
	reasoning := responses.ResponseReasoningItemParam{ID: strings.TrimSpace(meta.ID), Summary: reasoningSummaryParams(meta.Summary)}
	if encrypted := strings.TrimSpace(meta.EncryptedContent); encrypted != "" {
		reasoning.EncryptedContent = param.NewOpt(encrypted)
	}
	return responses.ResponseInputItemUnionParam{OfReasoning: &reasoning}, true
}

func decodeReasoningContinuation(v any) (openAIReasoningContinuation, bool) {
	if v == nil {
		return openAIReasoningContinuation{}, false
	}
	data, err := json.Marshal(v)
	if err != nil {
		return openAIReasoningContinuation{}, false
	}
	var meta openAIReasoningContinuation
	if err := json.Unmarshal(data, &meta); err != nil {
		return openAIReasoningContinuation{}, false
	}
	if strings.TrimSpace(meta.ID) == "" {
		return openAIReasoningContinuation{}, false
	}
	return meta, true
}

func reasoningSummaryText(summary []responses.ResponseReasoningItemSummary) string {
	texts := reasoningSummaryTexts(summary)
	return strings.Join(texts, "\n")
}

func reasoningSummaryTexts(summary []responses.ResponseReasoningItemSummary) []string {
	texts := make([]string, 0, len(summary))
	for _, item := range summary {
		if text := strings.TrimSpace(item.Text); text != "" {
			texts = append(texts, text)
		}
	}
	return texts
}

func reasoningSummaryParams(summary []string) []responses.ResponseReasoningItemSummaryParam {
	if len(summary) == 0 {
		return nil
	}
	out := make([]responses.ResponseReasoningItemSummaryParam, 0, len(summary))
	for _, text := range summary {
		if text = strings.TrimSpace(text); text != "" {
			out = append(out, responses.ResponseReasoningItemSummaryParam{Text: text})
		}
	}
	return out
}
