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
	"sort"
	"strings"

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

func (m *openAIModel) runStreaming(ctx context.Context, params responses.ResponseNewParams, yield func(*model.LLMResponse, error) bool) {
	stream := m.client.Responses.NewStreaming(ctx, params)
	defer stream.Close()

	state := newStreamState()

	for stream.Next() {
		evt := stream.Current()
		switch string(evt.Type) {
		case ResponseCompleted:
			state.captureResponse(&evt.Response)
		case ResponseFailed, ResponseIncomplete:
			state.captureResponse(&evt.Response)
		case Error:
			state.errorCode = evt.Code
			state.errorMessage = evt.Message
			if !yield(&model.LLMResponse{ErrorCode: evt.Code, ErrorMessage: evt.Message}, nil) {
				return
			}
		case ResponseOutputTextDelta:
			state.text.WriteString(evt.Delta)
			if evt.Delta != "" {
				if !yield(&model.LLMResponse{
					Partial:      true,
					TurnComplete: false,
					Content:      &genai.Content{Role: string(genai.RoleModel), Parts: []*genai.Part{{Text: evt.Delta}}},
				}, nil) {
					return
				}
			}
		case ResponseReasoningTextDelta, ResponseReasoningSummaryTextDelta:
			state.reasoning.WriteString(evt.Delta)
			if evt.Delta != "" {
				if !yield(&model.LLMResponse{
					Partial:      true,
					TurnComplete: false,
					Content:      &genai.Content{Role: string(genai.RoleModel), Parts: []*genai.Part{{Text: evt.Delta, Thought: true}}},
				}, nil) {
					return
				}
			}
		case ResponseRefusalDelta:
			state.text.WriteString(evt.Delta)
			if evt.Delta != "" {
				if !yield(&model.LLMResponse{
					Partial:      true,
					TurnComplete: false,
					Content:      &genai.Content{Role: string(genai.RoleModel), Parts: []*genai.Part{{Text: evt.Delta}}},
				}, nil) {
					return
				}
			}
		case ResponseFunctionCallArgumentsDelta:
			state.updateToolCall(evt.OutputIndex, "", "", evt.Delta, true)
		case ResponseFunctionCallArgumentsDone:
			state.updateToolCall(evt.OutputIndex, "", evt.Name, evt.Arguments, false)
		case ResponseOutputItemAdded, ResponseOutputItemDone:
			if string(evt.Item.Type) == FunctionCall {
				fc := evt.Item.AsFunctionCall()
				state.updateToolCall(evt.OutputIndex, fc.CallID, fc.Name, fc.Arguments, false)
			}
		}
	}

	if err := stream.Err(); err != nil {
		if ctx.Err() != nil {
			return
		}
		yield(&model.LLMResponse{ErrorCode: streamErrorCode, ErrorMessage: err.Error()}, nil)
		return
	}

	yield(state.finalResponse(), nil)
}

func (s *streamState) captureResponse(resp *responses.Response) {
	if resp == nil {
		return
	}
	s.response = resp
	if resp.ID != "" {
		s.responseID = resp.ID
	}
	if resp.Model != "" {
		s.model = string(resp.Model)
	}
	if resp.Status != "" {
		s.status = string(resp.Status)
		s.finishReason = string(resp.Status)
	}
	if resp.IncompleteDetails.Reason != "" {
		s.finishReason = resp.IncompleteDetails.Reason
	}
	if resp.Usage.InputTokens > 0 {
		s.promptTokens = resp.Usage.InputTokens
	}
	if resp.Usage.OutputTokens > 0 {
		s.completionTokens = resp.Usage.OutputTokens
	}
	if resp.Usage.TotalTokens > 0 {
		s.totalTokens = resp.Usage.TotalTokens
	}
	if resp.Error.Code != "" || resp.Error.Message != "" {
		s.errorCode = string(resp.Error.Code)
		s.errorMessage = resp.Error.Message
	}
}

type streamState struct {
	text             strings.Builder
	reasoning        strings.Builder
	response         *responses.Response
	toolCalls        map[int64]toolCallAccumulator
	finishReason     string
	promptTokens     int64
	completionTokens int64
	totalTokens      int64
	responseID       string
	model            string
	status           string
	errorCode        string
	errorMessage     string
}

type toolCallAccumulator struct {
	id        string
	name      string
	arguments strings.Builder
}

func newStreamState() *streamState {
	return &streamState{toolCalls: make(map[int64]toolCallAccumulator)}
}

func (s *streamState) updateToolCall(index int64, id, name, arguments string, appendArguments bool) {
	acc := s.toolCalls[index]
	if id != "" {
		acc.id = id
	}
	if name != "" {
		acc.name = name
	}
	if arguments != "" {
		if appendArguments {
			acc.arguments.WriteString(arguments)
		} else {
			acc.arguments.Reset()
			acc.arguments.WriteString(arguments)
		}
	}
	s.toolCalls[index] = acc
}

func (s *streamState) finalParts() []*genai.Part {
	if s.response != nil && len(s.response.Output) > 0 {
		if parts, _, hasContent := partsFromResponseOutput(s.response.Output); hasContent {
			return parts
		}
	}

	var parts []*genai.Part
	if s.reasoning.Len() > 0 {
		parts = append(parts, &genai.Part{Text: s.reasoning.String(), Thought: true})
	}
	if s.text.Len() > 0 {
		parts = append(parts, &genai.Part{Text: s.text.String()})
	}

	indices := make([]int64, 0, len(s.toolCalls))
	for index := range s.toolCalls {
		indices = append(indices, index)
	}
	sort.Slice(indices, func(i, j int) bool { return indices[i] < indices[j] })
	for _, index := range indices {
		acc := s.toolCalls[index]
		if acc.name == "" && acc.id == "" {
			continue
		}
		parts = append(parts, functionCallPart(acc.name, acc.id, acc.arguments.String()))
	}
	return parts
}

func (s *streamState) usageMetadata() *genai.GenerateContentResponseUsageMetadata {
	if s.promptTokens == 0 && s.completionTokens == 0 && s.totalTokens == 0 {
		return nil
	}
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(s.promptTokens),
		CandidatesTokenCount: int32(s.completionTokens),
		TotalTokenCount:      int32(s.totalTokens),
	}
}

func (s *streamState) finalResponse() *model.LLMResponse {
	resp := &model.LLMResponse{
		Partial:       false,
		TurnComplete:  true,
		FinishReason:  finishReasonFromString(s.finishReason),
		UsageMetadata: s.usageMetadata(),
		ModelVersion:  s.model,
		Content:       &genai.Content{Role: string(genai.RoleModel), Parts: s.finalParts()},
		ErrorCode:     s.errorCode,
		ErrorMessage:  s.errorMessage,
	}
	if s.responseID != "" || s.model != "" || s.status != "" {
		resp.CustomMetadata = make(map[string]any)
		if s.responseID != "" {
			resp.CustomMetadata[customMetadataResponseID] = s.responseID
		}
		if s.model != "" {
			resp.CustomMetadata[customMetadataModel] = s.model
		}
		if s.status != "" {
			resp.CustomMetadata[customMetadataStatus] = s.status
		}
	}
	return resp
}
