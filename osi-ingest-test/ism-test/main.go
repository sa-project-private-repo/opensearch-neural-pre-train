package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math/big"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/aws/aws-sdk-go-v2/config"
)

const (
	defaultEndpoint = "https://ism-logs-test-mwo3we5wqvdymi5s37uo77nd3y.us-east-1.osis.amazonaws.com"
	defaultPath     = "/logs"
	defaultRegion   = "us-east-1"
	serviceName     = "osis"
)

var (
	levels   = []string{"DEBUG", "INFO", "WARN", "ERROR", "FATAL"}
	sources  = []string{"api-gateway", "auth-service", "user-service", "payment-service", "order-service"}
	actions  = []string{"login", "logout", "create", "update", "delete", "read", "search", "export"}
	statuses = []string{"success", "failure", "pending", "timeout", "cancelled"}
)

type LogEntry struct {
	Timestamp  string `json:"timestamp"`
	Level      string `json:"level"`
	Source     string `json:"source"`
	Action     string `json:"action"`
	Status     string `json:"status"`
	UserID     string `json:"user_id"`
	RequestID  string `json:"request_id"`
	DurationMs int    `json:"duration_ms"`
	IP         string `json:"ip"`
	Message    string `json:"message"`
}

func randInt(max int) int {
	n, _ := rand.Int(rand.Reader, big.NewInt(int64(max)))
	return int(n.Int64())
}

func randChoice(arr []string) string {
	return arr[randInt(len(arr))]
}

func generateUUID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

func generateIP() string {
	return fmt.Sprintf("%d.%d.%d.%d", randInt(256), randInt(256), randInt(256), randInt(256))
}

func generateLog() LogEntry {
	level := randChoice(levels)
	action := randChoice(actions)
	status := randChoice(statuses)

	messages := map[string]string{
		"DEBUG":   fmt.Sprintf("Debug trace for %s operation", action),
		"INFO":    fmt.Sprintf("Successfully completed %s", action),
		"WARN":    fmt.Sprintf("Slow response detected during %s", action),
		"ERROR":   fmt.Sprintf("Failed to execute %s: connection timeout", action),
		"FATAL":   fmt.Sprintf("Critical failure in %s: service unavailable", action),
	}

	return LogEntry{
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		Level:      level,
		Source:     randChoice(sources),
		Action:     action,
		Status:     status,
		UserID:     fmt.Sprintf("user-%d", randInt(10000)+1),
		RequestID:  generateUUID(),
		DurationMs: randInt(5000) + 1,
		IP:         generateIP(),
		Message:    messages[level],
	}
}

func sendLogs(ctx context.Context, cfg aws.Config, endpoint, path string, logs []LogEntry) error {
	data, err := json.Marshal(logs)
	if err != nil {
		return fmt.Errorf("marshal error: %w", err)
	}

	url := endpoint + path
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("request creation error: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	// Calculate SHA256 hash of the payload
	h := sha256.Sum256(data)
	payloadHash := hex.EncodeToString(h[:])
	req.Header.Set("x-amz-content-sha256", payloadHash)

	creds, err := cfg.Credentials.Retrieve(ctx)
	if err != nil {
		return fmt.Errorf("credentials error: %w", err)
	}

	signer := v4.NewSigner()
	err = signer.SignHTTP(ctx, creds, req, payloadHash, serviceName, cfg.Region, time.Now())
	if err != nil {
		return fmt.Errorf("signing error: %w", err)
	}

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("request error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}

	return nil
}

func main() {
	endpoint := flag.String("endpoint", defaultEndpoint, "OSI endpoint URL")
	path := flag.String("path", defaultPath, "OSI path")
	region := flag.String("region", defaultRegion, "AWS region")
	interval := flag.Duration("interval", time.Second, "Send interval")
	batch := flag.Int("batch", 3, "Batch size")
	count := flag.Int("count", 0, "Total count (0=infinite)")
	flag.Parse()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\nShutting down...")
		cancel()
	}()

	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(*region))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load AWS config: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Starting log sender\n")
	fmt.Printf("  Endpoint: %s%s\n", *endpoint, *path)
	fmt.Printf("  Region:   %s\n", *region)
	fmt.Printf("  Interval: %v\n", *interval)
	fmt.Printf("  Batch:    %d\n", *batch)
	if *count > 0 {
		fmt.Printf("  Count:    %d\n", *count)
	} else {
		fmt.Printf("  Count:    infinite\n")
	}
	fmt.Println()

	ticker := time.NewTicker(*interval)
	defer ticker.Stop()

	sent := 0
	for {
		select {
		case <-ctx.Done():
			fmt.Printf("Total sent: %d documents\n", sent)
			return
		case <-ticker.C:
			logs := make([]LogEntry, *batch)
			for i := range logs {
				logs[i] = generateLog()
			}

			if err := sendLogs(ctx, cfg, *endpoint, *path, logs); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				continue
			}

			sent += *batch
			fmt.Printf("[%s] Sent %d docs (total: %d)\n",
				time.Now().Format("15:04:05"), *batch, sent)

			if *count > 0 && sent >= *count {
				fmt.Printf("Reached target count: %d\n", sent)
				return
			}
		}
	}
}
