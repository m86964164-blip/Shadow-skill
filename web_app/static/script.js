function startCam(){
    document.getElementById("video").src = "/video";
    fetch("/start");
}

function stopCam(){
    document.getElementById("video").src = "";
    fetch("/stop");
}

function downloadReport(){
    fetch("/generate_report")
    .then(r => r.json())
    .then(data => {
        alert("Report generated: " + data.file);
    });
}

setInterval(async () => {
    const res = await fetch("/score");
    const data = await res.json();

    document.getElementById("score").innerText = data.score + "%";
    document.getElementById("feedback").innerText = data.feedback;
}, 500);