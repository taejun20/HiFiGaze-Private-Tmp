setInterval(() => {
    fetch('/video_feed')
    .then(response => response.json())
    .then(data => {
        document.getElementById('status').innerText = data.button;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}, 500);