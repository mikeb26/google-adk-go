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

package fs

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"google.golang.org/adk/internal/sessionutils"
	"google.golang.org/adk/session"
)

// Option configures the filesystem-backed session service.
type Option func(*config)

type config struct {
	encoding Encoding
}

// WithEncoding sets the file encoding used for persisted sessions.
func WithEncoding(enc Encoding) Option {
	return func(cfg *config) {
		cfg.encoding = enc
	}
}

// NewSessionService creates a new [session.Service] implementation that stores
// sessions on the local filesystem under dir.
//
// The default encoding is JSON, but it can be overridden with [WithEncoding].
func NewSessionService(dir string, opts ...Option) (session.Service, error) {
	if strings.TrimSpace(dir) == "" {
		return nil, fmt.Errorf("dir is required")
	}

	cfg := config{encoding: JSONEncoding()}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	if cfg.encoding == nil {
		cfg.encoding = JSONEncoding()
	}
	if cfg.encoding.Ext() == "" {
		return nil, fmt.Errorf("encoding extension must not be empty")
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create session service directory: %w", err)
	}

	return &service{
		dir:      dir,
		encoding: cfg.encoding,
	}, nil
}

// service is a filesystem-backed implementation of session.Service.
type service struct {
	locks    sync.Map
	dir      string
	encoding Encoding
}

type persistedSession struct {
	AppName    string           `json:"app_name" yaml:"app_name"`
	UserID     string           `json:"user_id" yaml:"user_id"`
	SessionID  string           `json:"session_id" yaml:"session_id"`
	State      map[string]any   `json:"state" yaml:"state"`
	Events     []*session.Event `json:"events" yaml:"events"`
	CreateTime time.Time        `json:"create_time" yaml:"create_time"`
	UpdateTime time.Time        `json:"update_time" yaml:"update_time"`
}

type persistedState struct {
	State      map[string]any `json:"state" yaml:"state"`
	UpdateTime time.Time      `json:"update_time" yaml:"update_time"`
}

func (s *service) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	if req.AppName == "" || req.UserID == "" {
		return nil, fmt.Errorf("app_name and user_id are required, got app_name: %q, user_id: %q", req.AppName, req.UserID)
	}

	sessionID := req.SessionID
	if sessionID == "" {
		sessionID = uuid.NewString()
	}

	userUnlock, err := s.lockFile(s.userStatePath(req.AppName, req.UserID))
	if err != nil {
		return nil, err
	}
	defer userUnlock()

	appUnlock, err := s.lockFile(s.appStatePath(req.AppName))
	if err != nil {
		return nil, err
	}
	defer appUnlock()

	sessionUnlock, err := s.lockFile(s.sessionPath(req.AppName, req.UserID, sessionID))
	if err != nil {
		return nil, err
	}
	defer sessionUnlock()

	if _, err := s.loadPersistedSession(req.AppName, req.UserID, sessionID); err == nil {
		return nil, fmt.Errorf("session %s already exists", sessionID)
	} else if !errors.Is(err, fs.ErrNotExist) {
		return nil, err
	}

	appDelta, userDelta, sessionDelta := sessionutils.ExtractStateDeltas(req.State)
	now := time.Now().UTC().Truncate(time.Microsecond)
	appState, userState, err := s.applySharedStateDeltas(req.AppName, req.UserID, appDelta, userDelta, now)
	if err != nil {
		return nil, err
	}
	ps := &persistedSession{
		AppName:    req.AppName,
		UserID:     req.UserID,
		SessionID:  sessionID,
		State:      cloneMap(sessionDelta),
		Events:     []*session.Event{},
		CreateTime: now,
		UpdateTime: now,
	}
	if err := s.saveSession(ps); err != nil {
		return nil, err
	}

	val := newLocalSession(req.AppName, req.UserID, sessionID, ps.State, ps.Events, ps.UpdateTime)
	val.state = sessionutils.MergeStates(appState, userState, val.state)

	return &session.CreateResponse{Session: val}, nil
}

func (s *service) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		return nil, fmt.Errorf("app_name, user_id, session_id are required, got app_name: %q, user_id: %q, session_id: %q", req.AppName, req.UserID, req.SessionID)
	}

	userUnlock, err := s.lockFile(s.userStatePath(req.AppName, req.UserID))
	if err != nil {
		return nil, err
	}
	defer userUnlock()

	appUnlock, err := s.lockFile(s.appStatePath(req.AppName))
	if err != nil {
		return nil, err
	}
	defer appUnlock()

	sessionUnlock, err := s.lockFile(s.sessionPath(req.AppName, req.UserID, req.SessionID))
	if err != nil {
		return nil, err
	}
	defer sessionUnlock()

	sess, err := s.loadMergedSession(req.AppName, req.UserID, req.SessionID)
	if err != nil {
		return nil, err
	}

	filteredEvents := sess.events
	if req.NumRecentEvents > 0 {
		start := max(len(filteredEvents)-req.NumRecentEvents, 0)
		filteredEvents = filteredEvents[start:]
	}
	if !req.After.IsZero() && len(filteredEvents) > 0 {
		firstIndexToKeep := sort.Search(len(filteredEvents), func(i int) bool {
			return !filteredEvents[i].Timestamp.Before(req.After)
		})
		filteredEvents = filteredEvents[firstIndexToKeep:]
	}

	sess.events = cloneEvents(filteredEvents)
	return &session.GetResponse{Session: sess}, nil
}

func (s *service) List(ctx context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	if req.AppName == "" {
		return nil, fmt.Errorf("app_name is required, got app_name: %q", req.AppName)
	}

	appDir := s.appDir(req.AppName)
	if _, err := os.Stat(appDir); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return &session.ListResponse{Sessions: []session.Session{}}, nil
		}
		return nil, fmt.Errorf("failed to list sessions: %w", err)
	}

	type sessionCandidate struct {
		userID    string
		sessionID string
	}

	var candidates []sessionCandidate
	err := filepath.WalkDir(appDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			if d.Name() == ".state" {
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(d.Name()) != "."+s.encoding.Ext() {
			return nil
		}

		rel, err := filepath.Rel(appDir, path)
		if err != nil {
			return err
		}
		dir := filepath.Dir(rel)
		if dir == "." {
			return nil
		}
		parts := strings.Split(dir, string(filepath.Separator))
		if len(parts) != 1 || parts[0] == "" {
			return nil
		}

		userID, err := unsafeSegment(parts[0])
		if err != nil {
			return err
		}
		if req.UserID != "" && userID != req.UserID {
			return nil
		}

		sessionIDEncoded := strings.TrimSuffix(filepath.Base(path), "."+s.encoding.Ext())
		sessionID, err := unsafeSegment(sessionIDEncoded)
		if err != nil {
			return err
		}
		candidates = append(candidates, sessionCandidate{userID: userID, sessionID: sessionID})
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list sessions: %w", err)
	}

	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].userID != candidates[j].userID {
			return candidates[i].userID < candidates[j].userID
		}
		return candidates[i].sessionID < candidates[j].sessionID
	})

	var sessions []session.Session
	for _, cand := range candidates {
		userPath := s.userStatePath(req.AppName, cand.userID)
		appPath := s.appStatePath(req.AppName)
		sessionPath := s.sessionPath(req.AppName, cand.userID, cand.sessionID)

		userUnlock, err := s.lockFile(userPath)
		if err != nil {
			return nil, err
		}
		appUnlock, err := s.lockFile(appPath)
		if err != nil {
			userUnlock()
			return nil, err
		}
		sessionUnlock, err := s.lockFile(sessionPath)
		if err != nil {
			appUnlock()
			userUnlock()
			return nil, err
		}

		sess, err := s.loadMergedSession(req.AppName, cand.userID, cand.sessionID)
		if err != nil {
			sessionUnlock()
			appUnlock()
			userUnlock()
			if errors.Is(err, fs.ErrNotExist) {
				continue
			}
			return nil, err
		}
		sessions = append(sessions, sess)

		sessionUnlock()
		appUnlock()
		userUnlock()
	}

	sort.Slice(sessions, func(i, j int) bool {
		if sessions[i].AppName() != sessions[j].AppName() {
			return sessions[i].AppName() < sessions[j].AppName()
		}
		if sessions[i].UserID() != sessions[j].UserID() {
			return sessions[i].UserID() < sessions[j].UserID()
		}
		return sessions[i].ID() < sessions[j].ID()
	})

	return &session.ListResponse{Sessions: sessions}, nil
}

func (s *service) Delete(ctx context.Context, req *session.DeleteRequest) error {
	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		return fmt.Errorf("app_name, user_id, session_id are required, got app_name: %q, user_id: %q, session_id: %q", req.AppName, req.UserID, req.SessionID)
	}

	sessionUnlock, err := s.lockFile(s.sessionPath(req.AppName, req.UserID, req.SessionID))
	if err != nil {
		return err
	}
	defer sessionUnlock()

	path := s.sessionPath(req.AppName, req.UserID, req.SessionID)
	if err := os.Remove(path); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("failed to delete session: %w", err)
	}
	return nil
}

func (s *service) AppendEvent(ctx context.Context, curSession session.Session, event *session.Event) error {
	if curSession == nil {
		return fmt.Errorf("session is nil")
	}
	if event == nil {
		return fmt.Errorf("event is nil")
	}
	if event.Partial {
		return nil
	}

	event.Timestamp = event.Timestamp.UTC().Truncate(time.Microsecond)

	userUnlock, err := s.lockFile(s.userStatePath(curSession.AppName(), curSession.UserID()))
	if err != nil {
		return err
	}
	defer userUnlock()

	appUnlock, err := s.lockFile(s.appStatePath(curSession.AppName()))
	if err != nil {
		return err
	}
	defer appUnlock()

	sessionUnlock, err := s.lockFile(s.sessionPath(curSession.AppName(), curSession.UserID(), curSession.ID()))
	if err != nil {
		return err
	}
	defer sessionUnlock()

	ps, err := s.loadPersistedSession(curSession.AppName(), curSession.UserID(), curSession.ID())
	if err != nil {
		return err
	}
	if ps.UpdateTime.After(curSession.LastUpdateTime().UTC().Truncate(time.Microsecond)) {
		return fmt.Errorf(
			"stale session error: last update time from request (%s) is older than in database (%s)",
			curSession.LastUpdateTime().UTC().Format(time.RFC3339Nano),
			ps.UpdateTime.UTC().Format(time.RFC3339Nano),
		)
	}

	appDelta, userDelta, sessionDelta := sessionutils.ExtractStateDeltas(event.Actions.StateDelta)
	_, _, err = s.applySharedStateDeltas(curSession.AppName(), curSession.UserID(), appDelta, userDelta, event.Timestamp)
	if err != nil {
		return err
	}

	if sess, ok := curSession.(*localSession); ok {
		if err := sess.appendEvent(event); err != nil {
			return err
		}
		if len(sessionDelta) > 0 {
			maps.Copy(sess.state, sessionDelta)
		}
	} else {
		trimTempDeltaState(event)
	}

	ps.State = cloneMap(ps.State)
	if ps.State == nil {
		ps.State = make(map[string]any)
	}
	if len(sessionDelta) > 0 {
		maps.Copy(ps.State, sessionDelta)
	}
	ps.Events = append(ps.Events, cloneEvent(event))
	ps.UpdateTime = event.Timestamp
	if err := s.saveSession(ps); err != nil {
		return err
	}

	if sess, ok := curSession.(*localSession); ok {
		sess.updatedAt = ps.UpdateTime
	}
	return nil
}

func (s *service) appDir(app string) string {
	return filepath.Join(s.dir, safeSegment(app))
}

func (s *service) userDir(app, user string) string {
	return filepath.Join(s.appDir(app), safeSegment(user))
}

func (s *service) sessionPath(app, user, sessionID string) string {
	return filepath.Join(s.userDir(app, user), safeSegment(sessionID)+"."+s.encoding.Ext())
}

func (s *service) appStatePath(app string) string {
	return filepath.Join(s.appDir(app), ".state", "app."+s.encoding.Ext())
}

func (s *service) userStatePath(app, user string) string {
	return filepath.Join(s.userDir(app, user), ".state", "user."+s.encoding.Ext())
}

func (s *service) loadState(path string) (map[string]any, time.Time, error) {
	var state persistedState
	if err := s.load(path, &state); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return make(map[string]any), time.Time{}, nil
		}
		return nil, time.Time{}, err
	}
	if state.State == nil {
		state.State = make(map[string]any)
	}
	return state.State, state.UpdateTime, nil
}

func (s *service) loadPersistedSession(app, user, sessionID string) (*persistedSession, error) {
	var ps persistedSession
	if err := s.load(s.sessionPath(app, user, sessionID), &ps); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("session %s not found: %w", sessionID, fs.ErrNotExist)
		}
		return nil, err
	}
	if ps.State == nil {
		ps.State = make(map[string]any)
	}
	if ps.Events == nil {
		ps.Events = []*session.Event{}
	}
	return &ps, nil
}

func (s *service) loadMergedSession(app, user, sessionID string) (*localSession, error) {
	ps, appState, userState, err := s.loadSessionWithState(app, user, sessionID)
	if err != nil {
		return nil, err
	}

	sess := newLocalSession(app, user, sessionID, ps.State, ps.Events, ps.UpdateTime)
	sess.state = sessionutils.MergeStates(appState, userState, sess.state)
	return sess, nil
}

func (s *service) applySharedStateDeltas(app, user string, appDelta, userDelta map[string]any, updateTime time.Time) (map[string]any, map[string]any, error) {
	appState, _, err := s.loadState(s.appStatePath(app))
	if err != nil {
		return nil, nil, err
	}
	userState, _, err := s.loadState(s.userStatePath(app, user))
	if err != nil {
		return nil, nil, err
	}

	if len(appDelta) > 0 {
		maps.Copy(appState, appDelta)
		if err := s.saveState(s.appStatePath(app), persistedState{State: appState, UpdateTime: updateTime}); err != nil {
			return nil, nil, err
		}
	}
	if len(userDelta) > 0 {
		maps.Copy(userState, userDelta)
		if err := s.saveState(s.userStatePath(app, user), persistedState{State: userState, UpdateTime: updateTime}); err != nil {
			return nil, nil, err
		}
	}

	return appState, userState, nil
}

func (s *service) loadSessionWithState(app, user, sessionID string) (*persistedSession, map[string]any, map[string]any, error) {
	ps, err := s.loadPersistedSession(app, user, sessionID)
	if err != nil {
		return nil, nil, nil, err
	}

	appState, _, err := s.loadState(s.appStatePath(app))
	if err != nil {
		return nil, nil, nil, err
	}
	userState, _, err := s.loadState(s.userStatePath(app, user))
	if err != nil {
		return nil, nil, nil, err
	}
	return ps, appState, userState, nil
}

func (s *service) load(path string, out any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return err
		}
		return fmt.Errorf("failed to read %s: %w", path, err)
	}
	if err := s.encoding.Unmarshal(data, out); err != nil {
		return fmt.Errorf("failed to decode %s: %w", path, err)
	}
	return nil
}

func (s *service) saveState(path string, state persistedState) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("failed to create state directory: %w", err)
	}
	return s.save(path, state)
}

func (s *service) saveSession(ps *persistedSession) error {
	path := s.sessionPath(ps.AppName, ps.UserID, ps.SessionID)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("failed to create session directory: %w", err)
	}
	return s.save(path, ps)
}

func (s *service) save(path string, value any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	data, err := s.encoding.Marshal(value)
	if err != nil {
		return fmt.Errorf("failed to encode %s: %w", path, err)
	}

	tmp, err := os.CreateTemp(filepath.Dir(path), ".tmp-*"+filepath.Base(path))
	if err != nil {
		return fmt.Errorf("failed to create temp file for %s: %w", path, err)
	}
	tmpName := tmp.Name()
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
	}()

	if _, err := tmp.Write(data); err != nil {
		return fmt.Errorf("failed to write temp file for %s: %w", path, err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("failed to sync temp file for %s: %w", path, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("failed to close temp file for %s: %w", path, err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("failed to commit %s: %w", path, err)
	}
	if err := syncDir(filepath.Dir(path)); err != nil {
		return fmt.Errorf("failed to sync directory for %s: %w", path, err)
	}
	return nil
}

func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}

func safeSegment(seg string) string {
	escaped := url.PathEscape(seg)
	if escaped == "." {
		return "%2e"
	}
	if escaped == ".." {
		return "%2e%2e"
	}
	return escaped
}

func unsafeSegment(seg string) (string, error) {
	switch seg {
	case "%2e.":
		return ".", nil
	case "%2e..":
		return "..", nil
	default:
		decoded, err := url.PathUnescape(seg)
		if err != nil {
			return "", err
		}
		return decoded, nil
	}
}

func cloneMap(src map[string]any) map[string]any {
	if src == nil {
		return make(map[string]any)
	}
	return maps.Clone(src)
}

func cloneEvents(src []*session.Event) []*session.Event {
	if src == nil {
		return []*session.Event{}
	}
	return slicesClone(src)
}

func cloneEvent(event *session.Event) *session.Event {
	if event == nil {
		return nil
	}
	cloned := *event
	cloned.Actions.StateDelta = cloneMap(event.Actions.StateDelta)
	cloned.Actions.ArtifactDelta = maps.Clone(event.Actions.ArtifactDelta)
	cloned.Actions.RequestedToolConfirmations = maps.Clone(event.Actions.RequestedToolConfirmations)
	cloned.LongRunningToolIDs = slicesClone(event.LongRunningToolIDs)
	return &cloned
}

func slicesClone[T any](in []T) []T {
	if in == nil {
		return nil
	}
	out := make([]T, len(in))
	copy(out, in)
	return out
}
