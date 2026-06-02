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

import "errors"

var (
	ErrModelNameRequired              = errors.New("openai: model name is required")
	ErrClientRequired                 = errors.New("openai: client is required")
	ErrRequestNil                     = errors.New("openai: request is nil")
	ErrResponsesAPIFailed             = errors.New("openai: responses api failed")
	ErrNoContents                     = errors.New("openai: LLM request has no contents to convert")
	ErrStreamingUnavailable           = errors.New("openai: streaming unavailable")
	ErrFunctionCallMissingName        = errors.New("openai: function call missing name")
	ErrFunctionCallMissingID          = errors.New("openai: function call missing id")
	ErrFunctionResponseMissingName    = errors.New("openai: function response missing name")
	ErrFunctionResponseMissingID      = errors.New("openai: function response missing id")
	ErrToolCallMissingID              = errors.New("openai: tool call missing id")
	ErrToolCallMissingType            = errors.New("openai: tool call missing tool type")
	ErrToolResponseMissingID          = errors.New("openai: tool response missing id")
	ErrUnsupportedContentPart         = errors.New("openai: unsupported content part")
	ErrNilInlineData                  = errors.New("openai: nil inline data")
	ErrFileDataMissingURI             = errors.New("openai: file data is missing file uri")
	ErrNilTool                        = errors.New("openai: tool is nil")
	ErrNonFunctionToolUnsupported     = errors.New("openai: non-function tools are not supported")
	ErrFunctionDeclarationsRequired   = errors.New("openai: function tool must declare at least one function")
	ErrFunctionDeclarationMissingName = errors.New("openai: function declaration missing name")
	ErrUnsupportedToolCallingMode     = errors.New("openai: unsupported tool calling mode")
	ErrTopKNotSupported               = errors.New("openai: topK is not supported by the Responses API")
	ErrStopSequencesNotSupported      = errors.New("openai: stop sequences are not supported")
	ErrMultipleCandidatesNotSupported = errors.New("openai: multiple candidates per request are not supported")
	ErrPenaltiesNotSupported          = errors.New("openai: frequency/presence penalties are not supported")
	ErrLabelsNotSupported             = errors.New("openai: request labels are not supported")
	ErrSafetySettingsNotSupported     = errors.New("openai: gemini safety settings are not supported")
	ErrJSONResponseWithoutSchema      = errors.New("openai: json response requested without schema")
	ErrEmptyJSONSchema                = errors.New("openai: empty json schema")
	ErrUnsupportedResponseMIMEType    = errors.New("openai: unsupported response mime type")
	ErrEmptyResponse                  = errors.New("openai: empty response")
	ErrNoOutputItems                  = errors.New("openai: response included no output items")
	ErrNoTextOrToolContent            = errors.New("openai: response output did not contain text or tool content")
	ErrMarshalFunctionCallArguments   = errors.New("openai: marshal function call arguments")
	ErrMarshalToolCallArguments       = errors.New("openai: marshal tool call arguments")
)
