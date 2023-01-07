import "./styles/main.scss";
import Recorder from "./utils";

import chart from "./chart.png";

const btn = document.querySelector("#btn");

const ENDPOINT = "/analyze-parts-new";

let recorder;
let stream;
let audio_context;
let input;
let analyserNode;
let listening;
let requestId;

let recording;
let pending;
const threshold = 0.03;

let text;

// style
const ratio = 1.5;
const width = 500;

const initializeUserMedia = async function () {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audio_context = new AudioContext();
  input = audio_context.createMediaStreamSource(stream);
  recorder = new Recorder(input);
  analyserNode = audio_context.createAnalyser();
  input.connect(analyserNode);

  recording = false;
  pending = false;
  listening = false;
};

const startUserMedia = (stream) => {
  const pcmData = new Float32Array(analyserNode.fftSize);

  analyserNode.getFloatTimeDomainData(pcmData);
  let sumSquares = 0.0;
  for (const amplitude of pcmData) {
    sumSquares += amplitude * amplitude;
  }
  const volumeMeterEl = Math.sqrt(sumSquares / pcmData.length);

  if (volumeMeterEl >= threshold) {
    console.log(`>> ${volumeMeterEl}`);
  }

  // start recording
  if (volumeMeterEl >= threshold && recording === false && !pending) {
    recorder.record();
    recording = true;
  }

  if (volumeMeterEl < threshold && recording === true) {
    recorder.stop();
    recording = false;
    console.log("STOPPED");
    recorder.exportWAV((blob) => {
      let formData = new FormData();
      const file = new File([blob], "name_of_the_sound.wav", {
        type: "audio/wav",
        lastModified: Date.now(),
      });
      pending = true;
      formData.append("sound", file);
      fetch(ENDPOINT, {
        method: "POST",
        body: formData,
      })
        .then((res) => res.json())
        .then((data) => {
          console.log(data);
          text = data.prediction;
          const soundsElements = document.querySelectorAll(".sound");
          soundsElements.forEach((sound) => {
            if (sound.dataset.turkish === text) {
              sound.classList.add("active-sound");
            } else {
              sound.classList.remove("active-sound");
            }
          });
        })
        .finally(() => {
          createDownloadLink();
          recorder.clear();
          pending = false;
        });
    });
  }
};

const createDownloadLink = () => {
  if (recorder) {
    recorder.exportWAV(function (blob) {
      var url = URL.createObjectURL(blob);
      var li = document.createElement("li");
      var au = document.createElement("audio");
      var hf = document.createElement("a");

      au.controls = true;
      au.src = url;
      hf.href = url;
      hf.download = new Date().toISOString() + ".wav";
      // hf.innerHTML = hf.download;
      li.appendChild(au);
      li.appendChild(hf);

      const recordingslist = document.querySelector("#recordingslist");
      recordingslist.innerHTML = "";
      recordingslist.appendChild(li);
    });
  }
};

const onStart = () => {
  if (!requestId) {
    listening = true;
    requestId = window.requestAnimationFrame(loop);
  }
};

const onStop = () => {
  if (requestId) {
    window.cancelAnimationFrame(requestId);
    listening = false;
    requestId = null;
  }
};

const loop = () => {
  requestId = null;
  doStuff();
  onStart();
};

const doStuff = () => {
  startUserMedia(stream);
};

initializeUserMedia();

btn.addEventListener("click", () => {
  listening = !listening;

  if (listening) {
    btn.innerText = "Stop Listening";
    onStart();
  } else {
    btn.innerText = "Start Listening";
    onStop();
  }
});

const img = document.querySelector("#chart");
img.src = chart;

console.log(img.style.width);
console.log(getComputedStyle(img).width);

const soundsData = [
  {
    turkish: "i",
    soundLabel: "i",
    xpos: 45,
    ypos: 15,
  },
  {
    turkish: "ü",
    soundLabel: "y",
    xpos: 135,
    ypos: 45,
  },
  {
    turkish: "e",
    soundLabel: "e",
    xpos: 120,
    ypos: 135,
  },
  {
    turkish: "ö",
    soundLabel: "œ",
    xpos: 200,
    ypos: 135,
  },
  {
    turkish: "a",
    soundLabel: "a",
    xpos: 310,
    ypos: 325,
  },
  {
    turkish: "o",
    soundLabel: "o",
    xpos: 450,
    ypos: 125,
  },
  {
    turkish: "ı",
    soundLabel: "ɯ",
    xpos: 350,
    ypos: 15,
  },
  {
    turkish: "u",
    soundLabel: "u",
    xpos: 440,
    ypos: 15,
  },
];

const soundsElements = document.querySelectorAll(".sound");
soundsElements.forEach((sound) => {
  sound.addEventListener("webkitAnimationEnd", () => {
    sound.classList.remove("active-sound");
  });
  sound.addEventListener("animationend", () => {
    sound.classList.remove("active-sound");
  });

  const temp = soundsData.find((obj) => obj.turkish == sound.dataset.turkish);
  const chartImg = document.querySelector("#chart");
  chartImg.style.width = `${ratio * width}px`;

  sound.style.left = `${temp.xpos * ratio}px`;
  sound.style.top = `${temp.ypos * ratio}px`;
});
