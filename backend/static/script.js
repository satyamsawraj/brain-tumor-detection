async function handlePredict() {
    const fileInput = document.getElementById('imageInput');

    if (!fileInput.files.length) {
        alert("Upload image first");
        return;
    }

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    try {
        const res = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Show result
        document.getElementById("resultSection").style.display = "block";
        document.getElementById("predictionText").innerText = data.prediction;
        document.getElementById("confidenceText").innerText = data.confidence + "%";

        // Draw graph
        drawModelGraph(data.model_confidence);

        // Show best model
        document.getElementById("bestModelText").innerText =
            "🔥 Best Model for this MRI: " + data.best_model;

    } catch (err) {
        alert("Server error");
        console.error(err);
    }
}


function drawModelGraph(conf) {
    const ctx = document.getElementById("modelChart").getContext("2d");

    if (window.chart) {
        window.chart.destroy();
    }

    window.chart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["SVM", "META", "WOA"],
            datasets: [{
                label: "Confidence (%)",
                data: [conf.svm, conf.meta, conf.woa],
                backgroundColor: [
                    "#00c6ff",
                    "#00ff9f",
                    "#ff7675"
                ]
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}