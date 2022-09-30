function docReady(fn) {
        // see if DOM is already available
        if (document.readyState === "complete"
            || document.readyState === "interactive") {
            // call on next available tick
            setTimeout(fn, 1);
        } else {
            document.addEventListener("DOMContentLoaded", fn);
        }
    }

    docReady(function () {
        var resultContainer = document.getElementById('qr-reader-results');

        var lastResult, countResults = 0;
        function onScanSuccess(decodedText, decodedResult) {
            if (decodedText !== lastResult) {
                ++countResults;
                lastResult = decodedText;
                // Handle on success condition with the decoded message.
                console.log(`Scan result ${decodedText}`, decodedResult);
                
                resultContainer.innerHTML='User Data:'+decodedText+'\n'; //띄어쓰기 기준으로 나누기
                var data = decodedText.split(',');
                document.getElementById("id").value = data[0];
                document.getElementById("name").value = data[1];
                document.getElementById("location").value = data[2];
                document.getElementById("datetime").value = data[3];
                document.getElementById("concentration").value = data[4];
            }
        }

        var html5QrcodeScanner = new Html5QrcodeScanner(
            "qr-reader", { fps: 10, qrbox: 250 });
        html5QrcodeScanner.render(onScanSuccess);
    });