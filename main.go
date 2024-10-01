package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"time"

	"github.com/CyCoreSystems/audiosocket"
	"github.com/gorilla/websocket"
	"github.com/maxhawkins/go-webrtcvad"
	"github.com/pkg/errors"
)

// MaxCallDuration is the maximum amount of time to allow a call to be up before it is terminated.
const MaxCallDuration = 2 * time.Minute
const websocketURI = "ws://192.168.25.63:8001/ws"
const listenAddr = ":9092"
const languageCode = "en-US"

// slinChunkSize is the number of bytes which should be sent per Slin
// audiosocket message.  Larger data will be chunked into this size for
// transmission of the AudioSocket.
//
// This is based on 8kHz, 20ms, 16-bit signed linear.
const slinChunkSize = 320 // 8000Hz * 20ms * 2 bytes

func init() {
}

// ErrHangup indicates that the call should be terminated or has been terminated

// ErrHangup indicates that the call should be terminated or has been terminated
var ErrHangup = errors.New("Hangup")
var client = NewOllamaClient("http://192.168.25.63:11434")

func main() {
	var err error
	ctx := context.Background()

	client.AddSystemPrompt("You speak only on russian and you have a limit in one-two sentences and 180 symbols!")

	log.Println("listening for AudioSocket connections on", listenAddr)
	if err = Listen(ctx); err != nil {
		log.Fatalln("listen failure:", err)
	}
	log.Println("exiting")
}

func Listen(ctx context.Context) error {
	l, err := net.Listen("tcp", listenAddr)
	if err != nil {
		return errors.Wrapf(err, "failed to bind listener to socket %s", listenAddr)
	}

	for {
		conn, err := l.Accept()
		if err != nil {
			log.Println("failed to accept new connection:", err)
			continue
		}

		go Handle(ctx, conn)
	}
}

func Handle(pCtx context.Context, c net.Conn) {
	ctx, cancel := context.WithTimeout(pCtx, MaxCallDuration)
	defer cancel()
	defer c.Close()

	vad, err := webrtcvad.New()
	if err != nil {
		log.Fatal(err)
	}

	if err := vad.SetMode(3); err != nil {
		log.Fatal(err)
	}

	id, err := audiosocket.GetID(c)
	if err != nil {
		log.Println("failed to get call ID:", err)
		return
	}
	log.Printf("processing call %s", id.String())

	rate := 16000
	silenceThreshold := 5
	var inputAudioBuffer [][]float32
	var silenceCount int

	for ctx.Err() == nil {
		m, err := audiosocket.NextMessage(c)
		if errors.Cause(err) == io.EOF {
			log.Println("audiosocket closed")
			return
		}

		switch m.Kind() {
		case audiosocket.KindHangup:
			log.Println("audiosocket received hangup command")
			return
		case audiosocket.KindError:
			log.Println("error from audiosocket")
		case audiosocket.KindSlin:
			if m.ContentLength() < 1 {
				log.Println("no audio data")
				continue
			}
			audioData := m.Payload()

			floatArray, err := pcmToFloat32Array(audioData)
			if err != nil {
				log.Println("error converting pcm to float32:", err)
				continue
			}

			if active, err := vad.Process(rate, audioData); err != nil {
				log.Println("Error processing VAD:", err)
			} else if active {
				inputAudioBuffer = append(inputAudioBuffer, floatArray)
				silenceCount = 0
			} else {
				silenceCount++
				if silenceCount > silenceThreshold {
					if len(inputAudioBuffer) > 0 {
						log.Println("Processing complete sentence")
						handleInputAudio(c, inputAudioBuffer)
						inputAudioBuffer = nil // Reset buffer
					}
				}
			}

		}
	}
}
func handleInputAudio(conn net.Conn, buffer [][]float32) {
	// Merge and process buffer, then send to server
	var mergedBuffer []float32
	for _, data := range buffer {
		mergedBuffer = append(mergedBuffer, data...)
	}

	transcription, err := sendFloat32ArrayToServer("http://localhost:8002/complete_transcribe_r", mergedBuffer)
	if err != nil {
		log.Println("Error sending data to server:", err)
		return
	}
	res, errO := client.GenerateNoStream("gemma2:9b", transcription, nil, "")

	if errO != nil {
		fmt.Println("Error:", err)
	}
	log.Println("Received result:", res["response"])
	data := map[string]interface{}{
		"message":  res["response"],
		"language": "ru",
		"speed":    1.0,
	}
	log.Println("Using transcription:", transcription)
	websocketSendReceive(websocketURI, data, conn)

}

func pcmToFloat32Array(pcmData []byte) ([]float32, error) {
	if len(pcmData)%2 != 0 {
		return nil, fmt.Errorf("pcm data length must be even")
	}

	float32Array := make([]float32, len(pcmData)/2)
	buf := bytes.NewReader(pcmData)

	for i := 0; i < len(float32Array); i++ {
		var sample int16
		if err := binary.Read(buf, binary.LittleEndian, &sample); err != nil {
			return nil, fmt.Errorf("failed to read sample: %v", err)
		}
		float32Array[i] = float32(sample) / 32768.0
	}

	return float32Array, nil
}

func sendFloat32ArrayToServer(serverAddress string, float32Array []float32) (string, error) {
	var buf bytes.Buffer

	for _, f := range float32Array {
		if err := binary.Write(&buf, binary.LittleEndian, f); err != nil {
			return "", fmt.Errorf("failed to write float32: %v", err)
		}
	}

	req, err := http.NewRequest("POST", serverAddress, &buf)
	if err != nil {
		return "", fmt.Errorf("error creating request: %v", err)
	}
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("error sending request: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response body: %v", err)
	}

	log.Println("Server response:", string(body))
	//returh response ["transcription": message]
	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("error unmarshalling response body: %v", err)
	}

	if transcription, ok := result["transcription"].(string); ok {
		log.Println("Transcription:", transcription)
		return transcription, nil
		//return transcription
	} else {
		return "", fmt.Errorf("transcription not found in response")
	}
}

func websocketSendReceive(uri string, data map[string]interface{}, conn net.Conn) {
	wsConn, _, err := websocket.DefaultDialer.Dial(uri, nil)
	if err != nil {
		log.Println("Failed to connect to WebSocket:", err)
		return
	}
	defer wsConn.Close()

	err = wsConn.WriteJSON(data)
	if err != nil {
		log.Println("Failed to send JSON:", err)
		return
	}

	for {
		_, message, err := wsConn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("Unexpected WebSocket closure: %v", err)
			}
			break
		}

		var jsonMessage map[string]interface{}
		if err := json.Unmarshal(message, &jsonMessage); err == nil {
			if typeField, ok := jsonMessage["type"].(string); ok && typeField == "end_of_audio" {
				log.Println("End of conversation")
				break
			}
			log.Println("Received message:", jsonMessage)
		} else {
			// Assume non-JSON data can be audio bytes

			if _, err := conn.Write(audiosocket.SlinMessage(message)); err != nil {
				log.Println("Error writing to connection:", err)
				break
			}
		}
	}
}
