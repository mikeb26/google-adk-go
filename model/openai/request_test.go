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
	"testing"

	"google.golang.org/genai"
)

func TestContentsToResponsesInput_FunctionAndToolParts(t *testing.T) {
	contents := []*genai.Content{
		{
			Role: string(genai.RoleModel),
			Parts: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "lookup", Args: map[string]any{"city": "Paris"}}},
				{FunctionResponse: &genai.FunctionResponse{ID: "fc-1", Name: "lookup", Response: map[string]any{"temp": 72}}},
				{ToolCall: &genai.ToolCall{ID: "tc-1", ToolType: genai.ToolTypeURLContext, Args: map[string]any{"url": "https://example.com"}}},
				{ToolResponse: &genai.ToolResponse{ID: "tc-1", ToolType: genai.ToolTypeURLContext, Response: map[string]any{"result": "ok"}}},
			},
		},
	}

	input, _, err := contentsToResponsesInput(contents, nil)
	if err != nil {
		t.Fatalf("contentsToResponsesInput() error = %v", err)
	}
	if got, want := len(input.OfInputItemList), 4; got != want {
		t.Fatalf("unexpected item count: got %d want %d", got, want)
	}
	if input.OfInputItemList[0].OfFunctionCall == nil {
		t.Fatalf("expected first item to be a function call: %+v", input.OfInputItemList[0])
	}
	if input.OfInputItemList[1].OfFunctionCallOutput == nil {
		t.Fatalf("expected second item to be a function response: %+v", input.OfInputItemList[1])
	}
	if input.OfInputItemList[2].OfFunctionCall == nil {
		t.Fatalf("expected third item to be a tool call mapped as function call: %+v", input.OfInputItemList[2])
	}
	if input.OfInputItemList[3].OfFunctionCallOutput == nil {
		t.Fatalf("expected fourth item to be a tool response mapped as function output: %+v", input.OfInputItemList[3])
	}
}

func TestInlineDataToInputContent_UsesBlobDisplayName(t *testing.T) {
	blob := &genai.Blob{
		DisplayName: "foo.pdf",
		MIMEType:    applicationPDFMIMEType,
		Data:        []byte("pdf-bytes"),
	}

	content, err := inlineDataToInputContent(blob)
	if err != nil {
		t.Fatalf("inlineDataToInputContent() error = %v", err)
	}
	if content.OfInputFile == nil {
		t.Fatalf("expected input file content, got %+v", content)
	}
	if !content.OfInputFile.Filename.Valid() || content.OfInputFile.Filename.Value != "foo.pdf" {
		t.Fatalf("unexpected filename: %+v", content.OfInputFile.Filename)
	}
}

func TestInlineDataToInputContent_FallsBackToDefaultFilename(t *testing.T) {
	blob := &genai.Blob{
		MIMEType: applicationPDFMIMEType,
		Data:     []byte("pdf-bytes"),
	}

	content, err := inlineDataToInputContent(blob)
	if err != nil {
		t.Fatalf("inlineDataToInputContent() error = %v", err)
	}
	if content.OfInputFile == nil {
		t.Fatalf("expected input file content, got %+v", content)
	}
	if !content.OfInputFile.Filename.Valid() || content.OfInputFile.Filename.Value != defaultPDFFilename {
		t.Fatalf("unexpected filename: %+v", content.OfInputFile.Filename)
	}
}
