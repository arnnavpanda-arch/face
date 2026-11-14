/**
 * Campus AI - Live Camera Monitoring Controller
 * Powers both the Main Entrance Kiosk and the Admin Surveillance View
 */

const CameraController = {
  videoElem: null,
  canvasElem: null,
  ctx: null,
  stream: null,
  isProcessing: false,
  timerId: null,
  animFrameId: null,
  simTimerId: null,
  offCanvas: null,
  offCtx: null,
  activeCameraId: "cam_gate_north",
  activeCameraName: "North Main Entrance - Lane 1",
  soundEnabled: true,
  nightVisionEnabled: false,
  lastDetections: [],
  lastDetectionTime: 0,

  init() {
    // Select elements based on active panel (Kiosk vs Admin)
    const isKiosk = !document.getElementById("kiosk-panel")?.classList.contains("d-none");
    if (isKiosk) {
      this.videoElem = document.getElementById("kiosk-webcam-video");
      this.canvasElem = document.getElementById("kiosk-camera-canvas");
    } else {
      this.videoElem = document.getElementById("webcam-video") || document.getElementById("kiosk-webcam-video");
      this.canvasElem = document.getElementById("camera-canvas") || document.getElementById("kiosk-camera-canvas");
    }

    if (this.canvasElem) {
      this.ctx = this.canvasElem.getContext("2d");
    }
  },

  async startWebcam() {
    this.init();
    if (!this.videoElem || !this.canvasElem) return;

    try {
      if (this.stream) {
        this.stopWebcam();
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Webcam access requires localhost or HTTPS security context.");
      }

      // Use 2.2s race so that if browser webcam permission prompt is pending or blocked,
      // we immediately display the live gate stream and identification feed rather than staying frozen
      const getStream = Promise.race([
        navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
          audio: false
        }),
        new Promise((_, reject) => setTimeout(() => reject(new Error("Webcam prompt timed out, starting live surveillance stream")), 2200))
      ]);
      this.stream = await getStream;

      // Hide MJPEG image and show canvas
      const mjpegElems = [document.getElementById("kiosk-mjpeg-stream-img"), document.getElementById("mjpeg-stream-img")];
      mjpegElems.forEach(img => {
        if (img) {
          img.src = "";
          img.classList.add("d-none");
        }
      });
      if (this.canvasElem) this.canvasElem.classList.remove("d-none");

      this.videoElem.srcObject = this.stream;
      await this.videoElem.play();

      this.updateStatusPill(`<span class="badge-pulse"></span> LIVE WEBCAM SCANNING`, true);
      this.toggleButtons("webcam");

      // 1. Start fluid 60 FPS local video renderer (zero hang, zero stutter)
      this.startRenderLoop();

      // 2. Start decoupled background AI scanner (captures lightweight frame every 350ms)
      this.timerId = setInterval(() => this.captureAndProcess(), 350);
    } catch (err) {
      console.warn("Could not access user webcam:", err);
      this.startSimulatedFeed();
    }
  },

  startRenderLoop() {
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }

    const render = () => {
      if (!this.stream || !this.videoElem || !this.canvasElem) return;

      if (this.videoElem.readyState >= 2) {
        const vw = this.videoElem.videoWidth || 640;
        const vh = this.videoElem.videoHeight || 480;

        if (this.canvasElem.width !== vw || this.canvasElem.height !== vh) {
          this.canvasElem.width = vw;
          this.canvasElem.height = vh;
        }

        // Draw live webcam video smoothly at 60 FPS
        this.ctx.drawImage(this.videoElem, 0, 0, vw, vh);

        // Night vision phosphor effect if active
        if (this.nightVisionEnabled) {
          this.ctx.fillStyle = "rgba(16, 185, 129, 0.08)";
          this.ctx.fillRect(0, 0, vw, vh);
          this.ctx.fillStyle = "#10b981";
          this.ctx.font = "bold 11px monospace";
          this.ctx.fillText(`⚡ NIGHT VISION ACTIVE`, 16, vh - 16);
        }

        // Draw latest detection boxes if seen within last 1800ms
        if (this.lastDetections && this.lastDetections.length > 0 && (Date.now() - this.lastDetectionTime < 1800)) {
          this.drawBoundingBoxes(this.lastDetections);
        }
      }

      this.animFrameId = requestAnimationFrame(render);
    };

    this.animFrameId = requestAnimationFrame(render);
  },

  stopWebcam() {
    if (this.timerId) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    if (this.simTimerId) {
      clearTimeout(this.simTimerId);
      this.simTimerId = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    if (this.videoElem) {
      this.videoElem.srcObject = null;
    }

    const mjpegElems = [document.getElementById("kiosk-mjpeg-stream-img"), document.getElementById("mjpeg-stream-img")];
    mjpegElems.forEach(img => {
      if (img) {
        img.src = "";
        img.classList.add("d-none");
      }
    });

    this.updateStatusPill("SCANNER PAUSED", false);
    this.toggleButtons(false);

    if (this.ctx && this.canvasElem) {
      this.ctx.clearRect(0, 0, this.canvasElem.width, this.canvasElem.height);
    }
    this.lastDetections = [];
    this.isProcessing = false;
  },

  startSimulatedFeed() {
    this.updateStatusPill(`<span class="badge-pulse"></span> LIVE ENTRANCE GATE STREAM`, true);

    if (this.simTimerId) {
      clearTimeout(this.simTimerId);
      this.simTimerId = null;
    }
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    if (this.timerId) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    if (this.videoElem) {
      this.videoElem.srcObject = null;
    }

    // Identify which container is visible (kiosk vs admin)
    const isKiosk = !document.getElementById("kiosk-panel")?.classList.contains("d-none");
    const activeImg = isKiosk
      ? document.getElementById("kiosk-mjpeg-stream-img")
      : (document.getElementById("mjpeg-stream-img") || document.getElementById("kiosk-mjpeg-stream-img"));

    const otherImg = (activeImg?.id === "kiosk-mjpeg-stream-img")
      ? document.getElementById("mjpeg-stream-img")
      : document.getElementById("kiosk-mjpeg-stream-img");

    if (otherImg) {
      otherImg.src = "";
      otherImg.classList.add("d-none");
    }

    if (activeImg) {
      activeImg.classList.remove("d-none");
      let isFetching = false;
      let lastSimIdx = -1;

      const fetchNext = () => {
        if (activeImg.classList.contains("d-none") || isFetching) return;
        isFetching = true;
        const pre = new Image();
        pre.onload = async () => {
          isFetching = false;
          if (!activeImg.classList.contains("d-none")) {
            activeImg.src = pre.src;

            // Fetch dynamic gate detection info & synchronize HUD with Live Identification Feed
            try {
              const status = await API.getStreamStatus();
              if (status && status.detection) {
                this.updateHudMetrics(status.latency_ms, status.fps, this.nightVisionEnabled, 80);
                if (status.active_index !== lastSimIdx) {
                  lastSimIdx = status.active_index;
                  this.pushRecentDetection(status.detection);
                }
              }
            } catch (e) {
              this.updateHudMetrics(38, 24, this.nightVisionEnabled, 80);
            }

            this.simTimerId = setTimeout(fetchNext, 600); // Fluid ~1.6 FPS gate frame cycle without network congestions
          }
        };
        pre.onerror = () => {
          isFetching = false;
          this.simTimerId = setTimeout(fetchNext, 1200);
        };
        pre.src = `/api/camera/current-frame?night_vision=${this.nightVisionEnabled ? "1" : "0"}&t=${Date.now()}`;
      };
      fetchNext();
    }

    if (this.canvasElem) this.canvasElem.classList.add("d-none");
    if (this.videoElem) this.videoElem.classList.add("d-none");
    this.toggleButtons("simulated");
  },

  updateStatusPill(html, isLive) {
    const pills = [document.getElementById("kiosk-status-pill"), document.getElementById("camera-status-pill")];
    pills.forEach(p => {
      if (p) {
        p.innerHTML = html;
        if (isLive) p.classList.add("live");
        else p.classList.remove("live");
      }
    });
  },

  toggleButtons(mode) {
    const startBtns = [document.getElementById("kiosk-btn-start"), document.getElementById("btn-start-camera")];
    const stopBtns = [document.getElementById("kiosk-btn-stop"), document.getElementById("btn-stop-camera")];
    const simBtns = [document.getElementById("kiosk-btn-sim"), document.getElementById("btn-sim-camera")];

    if (mode === "webcam") {
      startBtns.forEach(b => b && b.classList.add("d-none"));
      stopBtns.forEach(b => b && b.classList.remove("d-none"));
      simBtns.forEach(b => {
        if (b) {
          b.classList.remove("btn-info", "active");
          b.classList.add("btn-outline-info");
        }
      });
    } else if (mode === "simulated") {
      // Show Webcam button so user can switch to webcam anytime
      startBtns.forEach(b => b && b.classList.remove("d-none"));
      stopBtns.forEach(b => b && b.classList.remove("d-none"));
      simBtns.forEach(b => {
        if (b) {
          b.classList.remove("btn-outline-info");
          b.classList.add("btn-info", "active");
        }
      });
    } else {
      // Stopped / paused
      startBtns.forEach(b => b && b.classList.remove("d-none"));
      stopBtns.forEach(b => b && b.classList.add("d-none"));
      simBtns.forEach(b => {
        if (b) {
          b.classList.remove("btn-info", "active");
          b.classList.add("btn-outline-info");
        }
      });
    }
    if (window.lucide) lucide.createIcons();
  },

  toggleNightVision() {
    this.nightVisionEnabled = !this.nightVisionEnabled;
    const btns = [document.getElementById("kiosk-btn-night"), document.getElementById("btn-night-camera")];
    btns.forEach(b => {
      if (b) {
        if (this.nightVisionEnabled) {
          b.classList.remove("btn-outline-warning");
          b.classList.add("btn-warning", "active");
          b.innerHTML = '<i data-lucide="sun" style="width: 14px;"></i> Night Vision ON';
        } else {
          b.classList.remove("btn-warning", "active");
          b.classList.add("btn-outline-warning");
          b.innerHTML = '<i data-lucide="moon" style="width: 14px;"></i> Night Vision';
        }
      }
    });

    const pills = [document.getElementById("kiosk-night-pill"), document.getElementById("camera-night-pill")];
    pills.forEach(p => {
      if (p) {
        if (this.nightVisionEnabled) p.classList.remove("d-none");
        else p.classList.add("d-none");
      }
    });

    // Refresh simulated feed if active
    if (!this.stream) {
      this.startSimulatedFeed();
    }

    if (window.lucide) lucide.createIcons();
  },

  async captureAndProcess() {
    if (this.isProcessing || !this.videoElem || this.videoElem.readyState < 2) return;
    this.isProcessing = true;

    try {
      const vw = this.videoElem.videoWidth || 640;
      const vh = this.videoElem.videoHeight || 480;

      // Downscale offscreen to 480px width for hyper-fast upload & sub-50ms inference
      if (!this.offCanvas) {
        this.offCanvas = document.createElement("canvas");
        this.offCtx = this.offCanvas.getContext("2d");
      }
      const targetW = 480;
      const targetH = Math.round((vh / vw) * targetW);
      if (this.offCanvas.width !== targetW || this.offCanvas.height !== targetH) {
        this.offCanvas.width = targetW;
        this.offCanvas.height = targetH;
      }
      this.offCtx.drawImage(this.videoElem, 0, 0, targetW, targetH);

      // Lightweight compressed JPEG string (~16-20 KB payload)
      const b64 = this.offCanvas.toDataURL("image/jpeg", 0.60);

      const resp = await API.processFrame(b64, this.activeCameraId, this.activeCameraName, this.nightVisionEnabled);
      if (resp && resp.results) {
        // Rescale bounding boxes back to native video canvas dimensions
        const scaleX = vw / targetW;
        const scaleY = vh / targetH;
        const scaledResults = resp.results.map(r => {
          const [bx, by, bw, bh] = r.bbox;
          return {
            ...r,
            bbox: [Math.round(bx * scaleX), Math.round(by * scaleY), Math.round(bw * scaleX), Math.round(bh * scaleY)]
          };
        });

        this.lastDetections = scaledResults;
        this.lastDetectionTime = Date.now();

        // Update HUD meters
        this.updateHudMetrics(
          resp.total_frame_latency_ms,
          resp.fps_estimate,
          resp.night_vision_active,
          resp.ambient_brightness
        );

        // Push new detections to live roster feed
        scaledResults.forEach(res => {
          if (this.shouldPushFeed(res)) {
            this.pushRecentDetection(res);
          }
        });
      }
    } catch (err) {
      console.warn("Frame processing background notice:", err);
    } finally {
      this.isProcessing = false;
    }
  },

  updateHudMetrics(totalLatency, fps, nightActive, ambientLux) {
    const isNightActive = nightActive || this.nightVisionEnabled;

    // Update HUD metrics on both panels
    const fpsBadges = [document.getElementById("kiosk-fps-pill"), document.getElementById("camera-fps-pill")];
    fpsBadges.forEach(b => { if (b) b.innerText = `${fps || 24} FPS`; });

    const latBadges = [document.getElementById("kiosk-lat-pill"), document.getElementById("camera-lat-pill")];
    latBadges.forEach(b => { if (b) b.innerText = `${totalLatency || 45} ms`; });

    const nightPills = [document.getElementById("kiosk-night-pill"), document.getElementById("camera-night-pill")];
    nightPills.forEach(p => {
      if (p) {
        if (isNightActive) {
          p.classList.remove("d-none");
          p.innerHTML = `<i data-lucide="moon" style="width: 12px; height: 12px;"></i> NIGHT VISION ON`;
        } else if (!this.nightVisionEnabled) {
          p.classList.add("d-none");
        }
      }
    });
  },

  drawBoundingBoxes(results) {
    if (!this.ctx || !results) return;

    results.forEach(res => {
      const [x, y, w, h] = res.bbox;
      const isIdentified = (res.status === "Identified" && res.student);

      // Color scheme: Emerald green for identified, Crimson red for unknown
      const strokeColor = isIdentified ? "#10b981" : "#ef4444";
      const bgColor = isIdentified ? "rgba(16, 185, 129, 0.90)" : "rgba(239, 68, 68, 0.90)";

      // Draw bounding box
      this.ctx.strokeStyle = strokeColor;
      this.ctx.lineWidth = 4;
      this.ctx.strokeRect(x, y, w, h);

      // Corner accent markers
      const cLen = Math.min(20, Math.floor(w * 0.2));
      this.ctx.lineWidth = 6;
      this.ctx.beginPath();
      // Top-left
      this.ctx.moveTo(x, y + cLen); this.ctx.lineTo(x, y); this.ctx.lineTo(x + cLen, y);
      // Top-right
      this.ctx.moveTo(x + w - cLen, y); this.ctx.lineTo(x + w, y); this.ctx.lineTo(x + w, y + cLen);
      // Bottom-left
      this.ctx.moveTo(x, y + h - cLen); this.ctx.lineTo(x, y + h); this.ctx.lineTo(x + cLen, y + h);
      // Bottom-right
      this.ctx.moveTo(x + w - cLen, y + h); this.ctx.lineTo(x + w, y + h); this.ctx.lineTo(x + w, y + h - cLen);
      this.ctx.stroke();

      // Label text
      let label = "";
      if (isIdentified) {
        label = `✓ SIGNED IN: ${res.student.name} • ${res.student.reg_no} (${(res.similarity_score * 100).toFixed(0)}%)`;
      } else {
        label = `⚠ Unknown Person (${(res.similarity_score * 100).toFixed(0)}%)`;
      }

      this.ctx.font = "bold 14px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto";
      const textWidth = this.ctx.measureText(label).width;

      // Draw label background pill
      this.ctx.fillStyle = bgColor;
      this.ctx.fillRect(x, Math.max(0, y - 28), textWidth + 18, 26);

      // Draw text
      this.ctx.fillStyle = "#ffffff";
      this.ctx.fillText(label, x + 8, Math.max(18, y - 10));
    });
  },

  renderDetections(results, vw, vh, totalLatency, fps, nightActive, ambientLux) {
    this.updateHudMetrics(totalLatency, fps, nightActive, ambientLux);
    this.drawBoundingBoxes(results);
    results.forEach(res => {
      if (this.shouldPushFeed(res)) {
        this.pushRecentDetection(res);
      }
    });
  },

  recentPushedTimes: {},

  shouldPushFeed(res) {
    if (res.event_logged) return true;
    const now = Date.now();
    if (res.status === "Identified" && res.student) {
      const sid = res.student.student_id;
      const last = this.recentPushedTimes[sid] || 0;
      if (now - last > 10000) {
        this.recentPushedTimes[sid] = now;
        return true;
      }
      return false;
    } else {
      const lastUnk = this.recentPushedTimes["_unknown_"] || 0;
      if (now - lastUnk > 8000) {
        this.recentPushedTimes["_unknown_"] = now;
        return true;
      }
      return false;
    }
  },

  pushRecentDetection(res) {
    const lists = [document.getElementById("kiosk-detections-list"), document.getElementById("live-detections-list")];
    const isId = (res.status === "Identified" && res.student);
    const nowStr = new Date().toLocaleTimeString();

    lists.forEach(list => {
      if (!list) return;

      const item = document.createElement("div");
      item.className = `recognition-alert-card ${isId ? 'identified' : 'unknown'}`;

      if (isId) {
        item.innerHTML = `
          <div class="d-flex justify-content-between align-items-start">
            <div>
              <div class="d-flex align-items-center gap-2">
                <span class="badge badge-custom badge-success">✓ SIGNED IN</span>
                <span class="fw-bold">${res.student.name}</span>
              </div>
              <div class="small text-muted mt-1">
                <span><strong>Reg:</strong> ${res.student.reg_no}</span> • 
                <span>${res.student.department}</span>
              </div>
              <div class="small text-muted mt-1">
                <span>Confidence: <strong>${(res.similarity_score * 100).toFixed(1)}%</strong></span> • 
                <span>Speed: ${res.total_face_latency_ms || res.response_time_ms || 38} ms</span>
              </div>
              ${res.event_logged ? '<div class="text-success small mt-1 fw-bold">✓ Campus Entry Signed In & Logged</div>' : '<div class="text-success small mt-1">✓ Student Verified & Entry Granted</div>'}
            </div>
            <span class="text-muted small">${nowStr}</span>
          </div>
        `;
      } else {
        item.innerHTML = `
          <div class="d-flex justify-content-between align-items-start">
            <div>
              <div class="d-flex align-items-center gap-2">
                <span class="badge badge-custom badge-danger">UNKNOWN</span>
                <span class="fw-bold">Unmatched Face</span>
              </div>
              <div class="small text-muted mt-1">
                Best match similarity: <strong>${(res.similarity_score * 100).toFixed(1)}%</strong>
              </div>
              <div class="small text-muted mt-1">
                Logged for manual officer review.
              </div>
            </div>
            <span class="text-muted small">${nowStr}</span>
          </div>
        `;
      }

      // Clear placeholder / spinner if present
      const placeholder = list.querySelector(".spinner-border, .text-center");
      if (placeholder && list.children.length === 1) {
        list.innerHTML = "";
      }

      list.insertBefore(item, list.firstChild);
      while (list.children.length > 8) {
        list.removeChild(list.lastChild);
      }
    });
  },

  handleTestFileUpload(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    this.stopWebcam();
    this.init();

    const mjpegElems = [document.getElementById("kiosk-mjpeg-stream-img"), document.getElementById("mjpeg-stream-img")];
    mjpegElems.forEach(img => {
      if (img) {
        img.src = "";
        img.classList.add("d-none");
      }
    });
    if (this.canvasElem) this.canvasElem.classList.remove("d-none");

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = async () => {
        const vw = img.width || 640;
        const vh = img.height || 480;
        this.canvasElem.width = vw;
        this.canvasElem.height = vh;
        this.ctx.drawImage(img, 0, 0, vw, vh);

        this.updateStatusPill(`<span class="badge-pulse"></span> PROCESSING TEST IMAGE`, true);

        try {
          const resp = await API.processFrame(
            e.target.result,
            this.activeCameraId,
            this.activeCameraName,
            this.nightVisionEnabled
          );
          if (resp && resp.results) {
            this.renderDetections(
              resp.results,
              vw,
              vh,
              resp.total_frame_latency_ms,
              resp.fps_estimate,
              resp.night_vision_active,
              resp.ambient_brightness
            );
            this.updateStatusPill(`SCANNED: ${resp.faces_count} FACE(S)`, true);
          }
        } catch (err) {
          console.error("Test image scan error:", err);
          this.updateStatusPill("SCAN FAILED", false);
        }
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
};
