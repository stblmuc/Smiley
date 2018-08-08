/* global $ */
class Main {
    constructor() {
        this.canvas = document.getElementById('main');
        this.input = document.getElementById('input');
        this.canvas.width = 449; // 16 * 28 + 1
        this.canvas.height = 449; // 16 * 28 + 1
        this.ctx = this.canvas.getContext('2d');
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.initialize();
    }

    initialize() {
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(0, 0, 449, 449);
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(0, 0, 449, 449);
        this.ctx.lineWidth = 0.05;
        for (var i = 0; i < 27; i++) {
            this.ctx.beginPath();
            this.ctx.moveTo((i + 1) * 16, 0);
            this.ctx.lineTo((i + 1) * 16, 449);
            this.ctx.closePath();
            this.ctx.stroke();

            this.ctx.beginPath();
            this.ctx.moveTo(0, (i + 1) * 16);
            this.ctx.lineTo(449, (i + 1) * 16);
            this.ctx.closePath();
            this.ctx.stroke();
        }
        this.drawInput((inputs)=>{})
        this.clearOutput();
    }

    onMouseDown(e) {
        this.canvas.style.cursor = 'default';
        this.drawing = true;
        this.prev = this.getPosition(e.clientX, e.clientY);
    }

    onMouseUp() {
        this.drawing = false;
        this.drawInput((inputs) => {
            this.loadOutput(inputs);
        });
    }

    onMouseMove(e) {
        if (this.drawing) {
            var curr = this.getPosition(e.clientX, e.clientY);
            this.ctx.lineWidth = 32;
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();
            this.ctx.moveTo(this.prev.x, this.prev.y);
            this.ctx.lineTo(curr.x, curr.y);
            this.ctx.stroke();
            this.ctx.closePath();
            this.prev = curr;
        }
    }

    getPosition(clientX, clientY) {
        var rect = this.canvas.getBoundingClientRect();
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }

    drawInput(cb) {
        var ctx = this.input.getContext('2d');
        var img = new Image();
        img.onload = () => {
            var inputs = [];
            var small = document.createElement('canvas').getContext('2d');
            small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28);
            var data = small.getImageData(0, 0, 28, 28).data;
            for (var i = 0; i < 28; i++) {
                for (var j = 0; j < 28; j++) {
                    var n = 4 * (i * 28 + j);
                    inputs[i * 28 + j] = (data[n + 0] + data[n + 1] + data[n + 2]);
                    ctx.fillStyle = 'rgb(' + [data[n + 0], data[n + 1], data[n + 2]].join(',') + ')';
                    ctx.fillRect(j * 5, i * 5, 5, 5);
                }
            }
            if (Math.min(...inputs) === 765) {
                return;
            }
            cb(inputs);
        };
        img.src = this.canvas.toDataURL();
    }

    clearOutput() {
        $("#error").text("");
        $('#output td, #output tr').remove();
    }

    loadOutput(inputs) {
        $.ajax({
            url: '/api/smiley',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(inputs),
            success: (data) => {

                const categories = data.categories;
                const classifiers = data.classifiers;
                const error = data.error;
                const results = data.results;

                if (error) {
                    $("#error").text(error);
                } else {
                    $("#error").text("");
                }

                // Don't display table if results contain empty arrays
                if (!results.filter((e)=>{return e.length}).length)
                    return;

                const table = $("#output");
                const thead = $("<thead>");
                const tbody = $("<tbody>");
                table.empty();
                table.append(thead);
                table.append(tbody);

                const headRow = $("<tr>");
                thead.append(headRow);
                headRow.append("<td>");
                for (let classifierIdx = 0; classifierIdx < classifiers.length; classifierIdx++) {
                    const cell = $("<th>");
                    headRow.append(cell);
                    cell.text(classifiers[classifierIdx]);
                }

                const mostSuccessfullCells = [];
                const mostSuccessfullValues = [];

                for (let categoryIdx = 0; categoryIdx < categories.length; categoryIdx++) {
                    const row = $("<tr>");
                    tbody.append(row);
                    const categoryNameCell = $("<td>");
                    row.append(categoryNameCell);
                    categoryNameCell.text(categories[categoryIdx]);
                    for (let classifierIdx = 0; classifierIdx < classifiers.length; classifierIdx++) {
                        const cell = $("<td>");
                        row.append(cell);
                        const result = results[classifierIdx][categoryIdx];
                        cell.text((result*100).toFixed(3)+"%");
                        const mostSuccessfullValue = mostSuccessfullValues[classifierIdx];
                        if (!mostSuccessfullValue || result > mostSuccessfullValue) {
                            mostSuccessfullValues[classifierIdx] = result;
                            mostSuccessfullCells[classifierIdx] = cell;
                        }
                    }
                }

                for (let index = 0; index < mostSuccessfullCells.length; index++){
                    mostSuccessfullCells[index].addClass("success");
                }
            }
        })
        .fail(() => {
            this.checkConnection();
        });
    }

    uploadTrainingData(inputs) {
        $.ajax({
            url: '/api/generate-training-example',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(inputs),
            success: (data) => {
                $("#trainigDataLabel").css("background-color", "#dff0d8");
                window.setTimeout(() => {
                    $("#trainigDataLabel").css("background-color", "#ffffff");
                }, 500);
                this.initialize();
            }
        })
        .fail(() => {
            this.checkConnection();
        });
    }

    loadImage(data) {
        var img = new Image();
        img.onload = () => {
            this.initialize();

            this.ctx.drawImage(img,0,0,449,449);
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(0, 0, 449, 449);

            this.drawInput((inputs) => {
                this.loadOutput(inputs);
            });
        }
        img.src = window.URL.createObjectURL(data)
    }

    takePicture(button) {
        if (!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
            if (!!this.video && !this.video.paused) {
                $(button).text("Camera");
                this.video.pause();

                this.drawInput((inputs)=>{
                    this.loadOutput(inputs);
                })
            } else if (!!this.video && this.video.paused) {
                $(button).text("Save");
                this.video.play();
            } else {
                var constraints = {video: {width: 449, height: 449, facingMode: "user", frameRate: 10}};

                navigator.mediaDevices.getUserMedia(constraints)
                .then((mediaStream) => {
                    $(button).text("Save!");

                    const ctx = this.ctx
                    this.video = document.createElement('video');
                    this.video.srcObject = mediaStream;
                    this.video.addEventListener('play', function(){
                        var $this = this;
                        (function loop() {
                            if (!$this.paused && !$this.ended) {
                                ctx.drawImage($this, 0, 0, 449, 449);
                                setTimeout(loop, 1000 / 10); // drawing at 10fps
                            }
                        })();
                    }, 0);
                    this.video.play();
                })
                .catch(function(err) { console.log(err.name + ": " + err.message); }); // always check for errors at the end.
            }
        } else {
            alert('getUserMedia() is not supported by your browser');
        }
    }

    trainModels(button) {
        $(button).prop('disabled', true);
        $(button).text("Training");
        $.ajax({
            url: '/api/train-models',
            method: 'POST',
            success: (data) => {
                this.drawInput((inputs) => {
                    this.loadOutput(inputs);
                });
            }
        })
        .always(() => {
            $(button).text("Train");
            $(button).prop('disabled', false);
        })
        .fail(() => {
            this.clearOutput();
            this.checkConnection();
        });
    }

    deleteAllModels(button) {
        if (!confirm("Do you really want to delete all trained models?"))
            return;

        $(button).prop('disabled', true);
        $.ajax({
            url: '/api/delete-all-models',
            method: 'POST',
            success: () => {
                this.clearOutput();
            }
        })
        .always(() => {
            $(button).prop('disabled', false);
        })
        .fail(() => {
            this.checkConnection();
        });
    }

    checkConnection() {
        const error = "<b>Please check the connection with the server.</b>";
        $("#error").html(error);
    }
}

$(() => {
    var main = new Main();
    $('#clear').click(() => {
        main.initialize();
    });

    $('#addTrainingData').click(() => {
        const label = $("#trainigDataLabel").val();
        if (label) {
            main.drawInput((inputs) => {
                const uploadData = {
                    cat: label,
                    img: inputs
                };
                main.uploadTrainingData(uploadData);
            });
        } else {
            alert("Please enter a name/label for the data");
        };
    });

    $('#importImage').change((e) => {
        main.loadImage(e.target.files[0]);
    });

    $('#takePicture').click((e) => {
        main.takePicture(e.target);
    });

    $('#trainModels').click((e) => {
        main.trainModels(e.target);
    });

    $('#deleteModels').click((e) => {
        main.deleteAllModels(e.target);
    });
});
