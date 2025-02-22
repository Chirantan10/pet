document.addEventListener("DOMContentLoaded", () => {
  const recordButton = document.getElementById("recordButton");
  const statusText = document.getElementById("status");
  let mediaRecorder;
  let audioChunks = [];
  let audioBlob;

  console.log("[DEBUG] DOM fully loaded.");

  // Request microphone access
  navigator.mediaDevices.getUserMedia({ audio: true })
      .then((stream) => {
          console.log("[DEBUG] Microphone access granted.");
          mediaRecorder = new MediaRecorder(stream);

          mediaRecorder.onstart = () => {
              console.log("[DEBUG] Recording started...");
              audioChunks = [];
              statusText.innerText = "Recording...";
          };

          mediaRecorder.ondataavailable = (event) => {
              console.log("[DEBUG] Audio data available.");
              if (event.data.size > 0) {
                  audioChunks.push(event.data);
              }
          };

          mediaRecorder.onstop = () => {
              console.log("[DEBUG] Audio recording stopped.");
              statusText.innerText = "Processing...";

              audioBlob = new Blob(audioChunks, { type: "audio/webm" });

              console.log("[DEBUG] Uploading recorded audio...");
              uploadAudio(audioBlob);
          };

          recordButton.addEventListener("click", () => {
              if (mediaRecorder.state === "inactive") {
                  mediaRecorder.start();
              } else {
                  mediaRecorder.stop();
              }
          });

      })
      .catch((error) => {
          console.error("[ERROR] Microphone access denied:", error);
          statusText.innerText = "Microphone access denied.";
      });

  function uploadAudio(audioBlob) {
      const formData = new FormData();
      formData.append("file", audioBlob, "audio.webm");

      fetch("http://127.0.0.1:5000/predict", {
          method: "POST",
          body: formData
      })
      .then(response => response.json())
      .then(data => {
          console.log("[DEBUG] Response received from backend:", data);

          if (data.error) {
              console.error("[ERROR] Backend error:", data.error);
              statusText.innerText = "Error: " + data.error;
              return;
          }

          // Display detected behavior
          statusText.innerText = "Detected behavior: " + data.behavior;

          // Auto-play response audio
          playResponseAudio();
      })
      .catch(error => {
          console.error("[ERROR] Error uploading audio:", error);
          statusText.innerText = "Upload failed!";
      });
  }

  function playResponseAudio() {
      console.log("[DEBUG] Playing response audio...");
      const audio = new Audio("http://127.0.0.1:5000/get_audio");
      audio.play()
          .then(() => console.log("[DEBUG] Audio playback started."))
          .catch(error => console.error("[ERROR] Audio playback failed:", error));
  }
});
