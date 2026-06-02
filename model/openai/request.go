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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

func contentsToResponsesInput(contents []*genai.Content, cfg *genai.GenerateContentConfig) (responses.ResponseNewParamsInputUnion, string, error) {
	instructions := systemInstructions(cfg)

	items := make(responses.ResponseInputParam, 0, len(contents))
	for _, content := range contents {
		if content == nil {
			continue
		}

		role := strings.ToLower(strings.TrimSpace(content.Role))
		if role == System {
			for _, part := range content.Parts {
				if part != nil && part.Text != "" {
					if instructions != "" {
						instructions += "\n"
					}
					instructions += part.Text
				}
			}
			continue
		}

		isAssistantLike := role == Assistant || role == Model

		var messageParts responses.ResponseInputMessageContentListParam
		var outputParts []responses.ResponseOutputMessageContentUnionParam

		for _, part := range content.Parts {
			if part == nil {
				continue
			}
			if reasoningInputItem, ok := reasoningInputItemFromPart(part); ok {
				if len(messageParts) > 0 {
					items = append(items, responses.ResponseInputItemParamOfMessage(messageParts, responsesRole(content.Role)))
					messageParts = nil
				}
				if len(outputParts) > 0 {
					items = append(items, responses.ResponseInputItemParamOfOutputMessage(
						outputParts,
						fmt.Sprintf("msg_adk_model_output_%d", len(items)),
						responses.ResponseOutputMessageStatusCompleted,
					))
					outputParts = nil
				}
				items = append(items, reasoningInputItem)
				continue
			}
			switch {
			case part.Text != "":
				if isAssistantLike {
					outputParts = append(outputParts, responses.ResponseOutputMessageContentUnionParam{
						OfOutputText: &responses.ResponseOutputTextParam{Text: part.Text},
					})
				} else {
					messageParts = append(messageParts, responses.ResponseInputContentUnionParam{
						OfInputText: &responses.ResponseInputTextParam{Text: part.Text},
					})
				}
			case part.InlineData != nil:
				if isAssistantLike {
					return responses.ResponseNewParamsInputUnion{}, "", fmt.Errorf("%w: assistant content cannot contain inline data", ErrUnsupportedContentPart)
				}
				inputPart, err := inlineDataToInputContent(part.InlineData)
				if err != nil {
					return responses.ResponseNewParamsInputUnion{}, "", err
				}
				messageParts = append(messageParts, inputPart)
			case part.FileData != nil:
				if isAssistantLike {
					return responses.ResponseNewParamsInputUnion{}, "", fmt.Errorf("%w: assistant content cannot contain file data", ErrUnsupportedContentPart)
				}
				inputPart, err := fileDataToInputContent(part.FileData)
				if err != nil {
					return responses.ResponseNewParamsInputUnion{}, "", err
				}
				messageParts = append(messageParts, inputPart)
			case part.FunctionCall != nil:
				if len(messageParts) > 0 {
					items = append(items, responses.ResponseInputItemParamOfMessage(messageParts, responsesRole(content.Role)))
					messageParts = nil
				}
				if len(outputParts) > 0 {
					items = append(items, responses.ResponseInputItemParamOfOutputMessage(
						outputParts,
						fmt.Sprintf("msg_adk_model_output_%d", len(items)),
						responses.ResponseOutputMessageStatusCompleted,
					))
					outputParts = nil
				}
				functionCall := part.FunctionCall
				if strings.TrimSpace(functionCall.ID) == "" {
					return responses.ResponseNewParamsInputUnion{}, "", ErrFunctionCallMissingID
				}
				if strings.TrimSpace(functionCall.Name) == "" {
					return responses.ResponseNewParamsInputUnion{}, "", ErrFunctionCallMissingName
				}
				argsJSON, err := json.Marshal(functionCall.Args)
				if err != nil {
					return responses.ResponseNewParamsInputUnion{}, "", fmt.Errorf("%w: %w", ErrMarshalFunctionCallArguments, err)
				}
				items = append(items, responses.ResponseInputItemParamOfFunctionCall(string(argsJSON), functionCall.ID, functionCall.Name))
			case part.FunctionResponse != nil:
				if len(messageParts) > 0 {
					items = append(items, responses.ResponseInputItemParamOfMessage(messageParts, responsesRole(content.Role)))
					messageParts = nil
				}
				if len(outputParts) > 0 {
					items = append(items, responses.ResponseInputItemParamOfOutputMessage(
						outputParts,
						fmt.Sprintf("msg_adk_model_output_%d", len(items)),
						responses.ResponseOutputMessageStatusCompleted,
					))
					outputParts = nil
				}
				functionResponse := part.FunctionResponse
				if strings.TrimSpace(functionResponse.Name) == "" {
					return responses.ResponseNewParamsInputUnion{}, "", ErrFunctionResponseMissingName
				}
				if strings.TrimSpace(functionResponse.ID) == "" {
					return responses.ResponseNewParamsInputUnion{}, "", ErrFunctionResponseMissingID
				}
				items = append(items, responses.ResponseInputItemParamOfFunctionCallOutput(functionResponse.ID, functionResponseContent(functionResponse)))
			case part.ToolCall != nil:
				if len(messageParts) > 0 {
					items = append(items, responses.ResponseInputItemParamOfMessage(messageParts, responsesRole(content.Role)))
					messageParts = nil
				}
				if len(outputParts) > 0 {
					items = append(items, responses.ResponseInputItemParamOfOutputMessage(
						outputParts,
						fmt.Sprintf("msg_adk_model_output_%d", len(items)),
						responses.ResponseOutputMessageStatusCompleted,
					))
					outputParts = nil
				}
				toolCall := part.ToolCall
				if strings.TrimSpace(toolCall.ID) == "" {
					return responses.ResponseNewParamsInputUnion{}, "", ErrToolCallMissingID
				}
				if toolCall.ToolType == genai.ToolTypeUnspecified {
					return responses.ResponseNewParamsInputUnion{}, "", ErrToolCallMissingType
				}
				argsJSON, err := json.Marshal(toolCall.Args)
				if err != nil {
					return responses.ResponseNewParamsInputUnion{}, "", fmt.Errorf("%w: %w", ErrMarshalToolCallArguments, err)
				}
				items = append(items, responses.ResponseInputItemParamOfFunctionCall(string(argsJSON), toolCall.ID, string(toolCall.ToolType)))
			case part.ToolResponse != nil:
				if len(messageParts) > 0 {
					items = append(items, responses.ResponseInputItemParamOfMessage(messageParts, responsesRole(content.Role)))
					messageParts = nil
				}
				if len(outputParts) > 0 {
					items = append(items, responses.ResponseInputItemParamOfOutputMessage(
						outputParts,
						fmt.Sprintf("msg_adk_model_output_%d", len(items)),
						responses.ResponseOutputMessageStatusCompleted,
					))
					outputParts = nil
				}
				toolResponse := part.ToolResponse
				if strings.TrimSpace(toolResponse.ID) == "" {
					return responses.ResponseNewParamsInputUnion{}, "", ErrToolResponseMissingID
				}
				items = append(items, responses.ResponseInputItemParamOfFunctionCallOutput(toolResponse.ID, toolResponseContent(toolResponse)))
			default:
				return responses.ResponseNewParamsInputUnion{}, "", fmt.Errorf("%w %T", ErrUnsupportedContentPart, part)
			}
		}
		if len(messageParts) > 0 {
			items = append(items, responses.ResponseInputItemParamOfMessage(messageParts, responsesRole(content.Role)))
		}
		if len(outputParts) > 0 {
			items = append(items, responses.ResponseInputItemParamOfOutputMessage(
				outputParts,
				fmt.Sprintf("msg_adk_model_output_%d", len(items)),
				responses.ResponseOutputMessageStatusCompleted,
			))
		}
	}

	if len(items) == 0 {
		items = append(items, responses.ResponseInputItemParamOfMessage(defaultResponseMessage, responses.EasyInputMessageRoleUser))
	}

	return responses.ResponseNewParamsInputUnion{OfInputItemList: items}, strings.TrimSpace(instructions), nil
}

func systemInstructions(cfg *genai.GenerateContentConfig) string {
	if cfg == nil || cfg.SystemInstruction == nil {
		return ""
	}
	var builder strings.Builder
	for _, part := range cfg.SystemInstruction.Parts {
		if part != nil && part.Text != "" {
			builder.WriteString(part.Text)
			builder.WriteByte('\n')
		}
	}
	return strings.TrimSpace(builder.String())
}

func responsesRole(role string) responses.EasyInputMessageRole {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case Assistant, Model:
		return responses.EasyInputMessageRoleAssistant
	case Developer:
		return responses.EasyInputMessageRoleDeveloper
	case System:
		return responses.EasyInputMessageRoleSystem
	default:
		return responses.EasyInputMessageRoleUser
	}
}

func inlineDataToInputContent(blob *genai.Blob) (responses.ResponseInputContentUnionParam, error) {
	if blob == nil {
		return responses.ResponseInputContentUnionParam{}, ErrNilInlineData
	}
	dataURL := dataURL(blob.MIMEType, blob.Data)
	if strings.HasPrefix(strings.ToLower(blob.MIMEType), imageMIMETypePrefix) {
		return responses.ResponseInputContentUnionParam{
			OfInputImage: &responses.ResponseInputImageParam{
				Detail:   responses.ResponseInputImageDetailAuto,
				ImageURL: param.NewOpt(dataURL),
			},
		}, nil
	}
	return responses.ResponseInputContentUnionParam{
		OfInputFile: &responses.ResponseInputFileParam{
			Filename: param.NewOpt(blobFilename(blob)),
			FileData: param.NewOpt(dataURL),
		},
	}, nil
}

func blobFilename(blob *genai.Blob) string {
	if blob == nil {
		return defaultInputFilename
	}
	if name := strings.TrimSpace(blob.DisplayName); name != "" {
		return name
	}
	return defaultFilename(blob.MIMEType)
}

func fileDataToInputContent(fileData *genai.FileData) (responses.ResponseInputContentUnionParam, error) {
	if fileData == nil {
		return responses.ResponseInputContentUnionParam{}, ErrNilInlineData
	}
	uri := strings.TrimSpace(fileData.FileURI)
	if uri == "" {
		return responses.ResponseInputContentUnionParam{}, ErrFileDataMissingURI
	}
	if strings.HasPrefix(strings.ToLower(fileData.MIMEType), imageMIMETypePrefix) {
		return responses.ResponseInputContentUnionParam{
			OfInputImage: &responses.ResponseInputImageParam{
				Detail:   responses.ResponseInputImageDetailAuto,
				ImageURL: param.NewOpt(uri),
			},
		}, nil
	}
	return responses.ResponseInputContentUnionParam{
		OfInputFile: &responses.ResponseInputFileParam{
			Filename: param.NewOpt(fileData.DisplayName),
			FileURL:  param.NewOpt(uri),
		},
	}, nil
}

func dataURL(mimeType string, data []byte) string {
	if mimeType == "" {
		mimeType = applicationOctetStreamMIMEType
	}
	return "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data)
}

func defaultFilename(mimeType string) string {
	switch strings.ToLower(mimeType) {
	case applicationPDFMIMEType:
		return defaultPDFFilename
	case textPlainMIMEType:
		return defaultTextFilename
	case textCSVMIMEType:
		return defaultCSVFilename
	default:
		return defaultInputFilename
	}
}

func functionResponseContent(resp *genai.FunctionResponse) string {
	if resp == nil {
		return ""
	}
	if len(resp.Parts) > 0 {
		if data, err := json.Marshal(resp.Parts); err == nil {
			return string(data)
		}
	}
	return responseContent(resp.Response)
}

func toolResponseContent(resp *genai.ToolResponse) string {
	if resp == nil {
		return ""
	}
	return responseContent(resp.Response)
}

func responseContent(resp any) string {
	if resp == nil {
		return ""
	}
	if s, ok := resp.(string); ok {
		return s
	}
	if m, ok := resp.(map[string]any); ok {
		if result, ok := m[mapKeyResult].(string); ok {
			return result
		}
		if output, ok := m[mapKeyOutput].(string); ok {
			return output
		}
		if content, ok := m[mapKeyContent].(string); ok {
			return content
		}
	}
	data, _ := json.Marshal(resp)
	return string(data)
}

func genaiToolsToResponses(tools []*genai.Tool) ([]responses.ToolUnionParam, error) {
	var out []responses.ToolUnionParam
	for _, t := range tools {
		if t == nil {
			return nil, ErrNilTool
		}
		if t.Retrieval != nil || t.GoogleSearch != nil || t.GoogleSearchRetrieval != nil || t.GoogleMaps != nil ||
			t.EnterpriseWebSearch != nil || t.URLContext != nil || t.ComputerUse != nil || t.CodeExecution != nil ||
			t.FileSearch != nil || t.MCPServers != nil || t.ParallelAISearch != nil {
			return nil, ErrNonFunctionToolUnsupported
		}
		if len(t.FunctionDeclarations) == 0 {
			return nil, ErrFunctionDeclarationsRequired
		}
		for _, fd := range t.FunctionDeclarations {
			if fd == nil {
				return nil, ErrFunctionDeclarationMissingName
			}
			if fd.Name == "" {
				return nil, ErrFunctionDeclarationMissingName
			}
			out = append(out, responses.ToolUnionParam{
				OfFunction: &responses.FunctionToolParam{
					Name:        fd.Name,
					Description: param.NewOpt(fd.Description),
					Parameters:  functionParameters(fd),
					Strict:      param.NewOpt(true),
				},
			})
		}
	}
	return out, nil
}

func functionParameters(fd *genai.FunctionDeclaration) map[string]any {
	if fd == nil {
		return map[string]any{mapKeyType: schemaObjectType, mapKeyProperties: map[string]any{}}
	}
	if fd.ParametersJsonSchema != nil {
		return schemaMap(fd.ParametersJsonSchema)
	}
	if fd.Parameters != nil {
		return schemaMap(fd.Parameters)
	}
	return map[string]any{mapKeyType: schemaObjectType, mapKeyProperties: map[string]any{}}
}

func schemaMap(schema any) map[string]any {
	paramsMap := make(map[string]any)
	switch m := schema.(type) {
	case nil:
	case map[string]any:
		maps.Copy(paramsMap, m)
	case map[string]string:
		for k, v := range m {
			paramsMap[k] = v
		}
	default:
		data, err := json.Marshal(schema)
		if err == nil {
			var decoded map[string]any
			if json.Unmarshal(data, &decoded) == nil {
				maps.Copy(paramsMap, decoded)
			}
		}
	}
	if _, ok := paramsMap[mapKeyType]; !ok {
		paramsMap[mapKeyType] = schemaObjectType
	}
	if paramsMap[mapKeyType] == nil || paramsMap[mapKeyType] == "" {
		paramsMap[mapKeyType] = schemaObjectType
	}
	if paramsMap[mapKeyType] == schemaObjectType {
		if _, ok := paramsMap[mapKeyProperties]; !ok {
			paramsMap[mapKeyProperties] = map[string]any{}
		}
	}
	return paramsMap
}
