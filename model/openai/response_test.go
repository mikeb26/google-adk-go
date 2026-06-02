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
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/adk/model"
)

func TestLLMResponseFromResponse(t *testing.T) {
	var resp responses.Response
	if err := json.Unmarshal([]byte(`{
		"id":"resp_1",
		"model":"gpt-4o-mini",
		"status":"completed",
		"output":[{"type":"message","id":"msg_1","status":"completed","content":[{"type":"output_text","text":"hello"}]}],
		"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}
	}`), &resp); err != nil {
		t.Fatalf("failed to unmarshal response fixture: %v", err)
	}

	llmResp, err := llmResponseFromResponse(&resp)
	if err != nil {
		t.Fatalf("llmResponseFromResponse() error = %v", err)
	}
	if llmResp == nil || llmResp.Content == nil || len(llmResp.Content.Parts) != 1 || llmResp.Content.Parts[0].Text != "hello" {
		t.Fatalf("unexpected response: %+v", llmResp)
	}
	if llmResp.CustomMetadata[customMetadataResponseID] != "resp_1" {
		t.Fatalf("expected metadata response id, got %+v", llmResp.CustomMetadata)
	}
}

var _ model.LLMResponse
