

const video = document.getElementById("videoInput");

console.log(faceapi.nets);

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
]).then(start);

function start() {
  document.body.append("models loaded");
  video.src = "../videos/avengers.mp4";

  recongnizeFaces();
}

async function recongnizeFaces() {
  const labeledDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6); //load distance=0.6

  video.addEventListener("play", () => {
    console.log("playing");

    const canvas = faceapi.createCanvasFromMedia(video);
    document.body.append(canvas);
    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(canvas, displaySize);

    setInterval(async () => {
      const detections = await faceapi
        .detectAllFaces(video)
        .withFaceLandmarks()
        .withFaceDescriptors();

      const resizedDetections = faceapi.resizeResults(detections, displaySize);
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
      const results = resizedDetections.map((d) => {
        return faceMatcher.findBestMatch(d.descriptor);
      });
      results.forEach((result, i) => {
        const box = resizedDetections[i].detection.box;
        const drawBox = new faceapi.draw.DrawBox(box, {
          label: result.toString(),
        });
        drawBox.draw(canvas);
      });
    }, 100);
  });
}

function loadLabeledImages() {
  const labels = ["black widow", "iron man"];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(
          `./labeled_images/${label}/${i}.jpg`
        ) //burası yanlış olabilir
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor()
        descriptions.push(detections.descriptor)
      }
      document.body.append(label + "faces loaded");
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
