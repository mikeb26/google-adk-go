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
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/gofrs/flock"
)

const lockFileSuffix = ".lck"

func (s *service) lockFile(path string) (func(), error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("failed to create lock directory for %s: %w", path, err)
	}

	lockPath := path + lockFileSuffix
	mu := s.getMutex(lockPath)
	mu.Lock()

	fileLock := flock.New(lockPath)
	if err := fileLock.Lock(); err != nil {
		mu.Unlock()
		return nil, fmt.Errorf("failed to acquire lock for %s: %w", path, err)
	}

	return func() {
		_ = fileLock.Unlock()
		mu.Unlock()
	}, nil
}

func (s *service) getMutex(key string) *sync.Mutex {
	if v, ok := s.locks.Load(key); ok {
		return v.(*sync.Mutex)
	}

	mu := &sync.Mutex{}
	actual, _ := s.locks.LoadOrStore(key, mu)
	return actual.(*sync.Mutex)
}
