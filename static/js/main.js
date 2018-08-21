/* global $ */
class Main {
    constructor() {
        this.canvas = document.getElementById('main');
        this.ctx = this.canvas.getContext('2d');
        this.input = document.getElementById('input');  

        this.numAugm = param.numAugm;
        this.batchSize = param.batchSize;
        this.lrRate = param.lrRate;
        this.lrEpochs = param.lrEpochs;
        this.cnnRate = param.cnnRate;
        this.cnnEpochs = param.cnnEpochs;
        this.image_size = param.image_size;
        this.rect_size = 448; // 16 * 28
        this.col_width = this.rect_size / this.image_size; // for the grid

        this.canvas.width = this.rect_size;
        this.canvas.height = this.rect_size;
        this.input.width = 5 * this.image_size;
        this.input.height = 5 * this.image_size;

        var catsList = document.getElementById('trainingDataLabelOptions');
        this.cats = param.categories;
        this.cats.forEach(function(item){
            var option = document.createElement('option');
            option.value = item;
            catsList.appendChild(option);
        });

        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));

        this.initializeConfigValues();
        this.initialize();
    }

    initialize() {
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.input.getContext('2d').clearRect(0,0, this.input.width, this.input.height);

        this.ctx.lineWidth = 0.05;
        for (var i = 0; i < this.image_size; i++) {
            this.ctx.beginPath();
            this.ctx.moveTo((i + 1) * this.col_width, 0);
            this.ctx.lineTo((i + 1) * this.col_width, this.rect_size);
            this.ctx.closePath();
            this.ctx.stroke();

            this.ctx.beginPath();
            this.ctx.moveTo(0, (i + 1) * this.col_width);
            this.ctx.lineTo(this.rect_size, (i + 1) * this.col_width);
            this.ctx.closePath();
            this.ctx.stroke();
        }
        this.clearOutput();
    }

    initializeConfigValues() {
        document.getElementById('num-augm').value = this.numAugm;
        document.getElementById('batch-size').value = this.batchSize;
        document.getElementById('lr-rate').value = this.lrRate;
        document.getElementById('lr-epochs').value = this.lrEpochs;
        document.getElementById('cnn-rate').value = this.cnnRate;
        document.getElementById('cnn-epochs').value = this.cnnEpochs;
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

        if (!!this.video && !this.video.paused) $('#takePicture').click();
    }

    onMouseMove(e) {
        if (this.drawing) {
            var curr = this.getPosition(e.clientX, e.clientY);
            this.ctx.lineWidth = Math.max(5, 46 - (this.image_size / 2));
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
            small.drawImage(img, 0, 0, img.width, img.height, 0, 0, this.image_size, this.image_size);
            var data = small.getImageData(0, 0, this.image_size, this.image_size).data;

            // get max and min for normalization
            var max = 0
            var min = 0
            for (var i = 0; i < this.image_size; i++) {
                for (var j = 0; j < this.image_size; j++) {
                    var n = 4 * (i * this.image_size + j);
                    var grayscale = (data[n + 0]*.3 + data[n + 1]*.59 + data[n + 2]*.11)
                    max = Math.max(max,grayscale)
                    min = Math.min(min,grayscale)
                }
            }

            for (var i = 0; i < this.image_size; i++) {
                for (var j = 0; j < this.image_size; j++) {
                    var n = 4 * (i * this.image_size + j);
                    var grayscale = (data[n + 0]*.3 + data[n + 1]*.59 + data[n + 2]*.11)
                    grayscale = 255 * (grayscale - min) / (max - min)

                    // Threshold
                    const threshold = 51
                    const scale = 3
                    var scaled_gray = Math.min(255,((grayscale - threshold)*scale) + threshold)
                    grayscale = grayscale > threshold ? scaled_gray : grayscale

                    inputs[i * this.image_size + j] = grayscale;
                    ctx.fillStyle = 'rgb(' + Array(3).fill(grayscale) + ')';
                    ctx.fillRect(j * 5, i * 5, 5, 5);
                }
            }
            if (Math.min(...inputs) === 255) {
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
                headRow.append("<th>");
                for (let classifierIdx = 0; classifierIdx < classifiers.length; classifierIdx++) {
                    const cell = $("<th scope='col'>");
                    headRow.append(cell);
                    cell.text(classifiers[classifierIdx]);
                }

                const mostSuccessfulCells = [];
                const mostSuccessfulValues = [];

                for (let categoryIdx = 0; categoryIdx < categories.length; categoryIdx++) {
                    const row = $("<tr>");
                    tbody.append(row);
                    const categoryNameCell = $("<th scope='row'>");
                    row.append(categoryNameCell);
                    categoryNameCell.text(categories[categoryIdx]);
                    for (let classifierIdx = 0; classifierIdx < classifiers.length; classifierIdx++) {
                        const cell = $("<td>");
                        row.append(cell);
                        const result = results[classifierIdx][categoryIdx];
                        cell.text((result*100).toFixed(3)+"%");
                        const mostSuccessfulValue = mostSuccessfulValues[classifierIdx];
                        if (!mostSuccessfulValue || result > mostSuccessfulValue) {
                            mostSuccessfulValues[classifierIdx] = result;
                            mostSuccessfulCells[classifierIdx] = cell;
                        }
                    }
                }

                for (let index = 0; index < mostSuccessfulCells.length; index++){
                    mostSuccessfulCells[index].addClass("table-success");
                }
            }
        })
        .fail(() => {
            this.clearOutput();
            this.checkConnection();
        });
    }

    addTrainingData(button) {
        const label = $("#trainingDataLabel").val();
        if (label) {
            this.drawInput((inputs) => {
                const uploadData = {
                    cat: label,
                    img: inputs
                };
                this.uploadTrainingData(uploadData);
            });
            if (!this.cats.includes(label)) {
                this.cats.push(label)
                var catsList = document.getElementById('trainingDataLabelOptions');
                var option = document.createElement('option');
                option.value = label;
                catsList.appendChild(option);
            }
        } else {
            alert("Please enter a name/label for the data");
        };
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
            this.clearOutput();
            this.checkConnection();
        });
    }

    loadImage(data, cb) {
        var img = new Image();
        img.onload = () => {
            this.initialize();
            var imgSize = Math.min(img.width, img.height);
    	    var left = (img.width - imgSize) / 2;
    	    var top = (img.height - imgSize) / 2;

            // draw squared-up image in canvas
            this.ctx.drawImage(img, left, top, imgSize, imgSize, 0, 0, this.ctx.canvas.width, this.ctx.canvas.height);

            this.drawInput((inputs) => {
                if (typeof cb == 'function')
                    cb(data, inputs);
                else
                    this.loadOutput(inputs);
            });
        }
        img.src = window.URL.createObjectURL(data)
    }

    loadAndUploadImages(target) {
        function cb(data, inputs) {
            var path = data.webkitRelativePath.split("/");
            var label = path[path.length - 2];
            if (label) {
                const uploadData = {
                    cat: label,
                    img: inputs
                };
                $.ajax({
                    url: '/api/generate-training-example',
                    method: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify(uploadData),
                    success: (data) => {
                    }
                })
            } else {
                alert("Please select a folder of one category or of one image size");
            };
        }

        for (var i = 0; i < target.files.length; i++) {
            this.loadImage(target.files[i], cb);
        }
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
                var constraints = {video: {width: this.rect_size, height: this.rect_size, facingMode: "user", frameRate: 10}};

                navigator.mediaDevices.getUserMedia(constraints)
                .then((mediaStream) => {
                    $(button).text("Save");

                    const ctx = this.ctx
                    const rect_size = this.rect_size

                    this.video = document.createElement('video');
                    this.video.srcObject = mediaStream;
                    this.video.addEventListener('play', function(){
                        var $this = this;
                        (function loop() {
                            if (!$this.paused && !$this.ended) {
                                ctx.drawImage($this, 0, 0, rect_size, rect_size);
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
        $("#deleteModels").prop('disabled', true);
        var blink = setInterval(function(){
            $(button).fadeOut(400).fadeIn(400);
        }, 1000);

        $.ajax({
            url: '/api/train-models',
            method: 'POST',
            success: (data) => {
                const error = data.error;
                if (error) {
                    $("#error").text(error);
                } else {
                    $("#error").text("");
                    this.drawInput((inputs) => {
                        this.loadOutput(inputs);
                    });
                }
            }
        })
        .always(() => {
            clearInterval(blink);
            $(button).prop('disabled', false);
            $("#deleteModels").prop('disabled', false);
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
        var blink = setInterval(function(){
            $(button).fadeOut(400).fadeIn(400);
        }, 1000);

        $.ajax({
            url: '/api/delete-all-models',
            method: 'POST',
            success: () => {
                this.clearOutput();
            }
        })
        .always(() => {
            clearInterval(blink);
            $(button).prop('disabled', false);
        })
        .fail(() => {
            this.clearOutput();
            this.checkConnection();
        });
    }

    updateConfig(button) {
        this.numAugm = document.getElementById('num-augm').value;
        this.batchSize = document.getElementById('batch-size').value;
        this.lrRate = document.getElementById('lr-rate').value;
        this.lrEpochs = document.getElementById('lr-epochs').value;
        this.cnnRate = document.getElementById('cnn-rate').value;
        this.cnnEpochs = document.getElementById('cnn-epochs').value;

        const conf = {
            numberAugmentations: this.numAugm,
            batchSize: this.batchSize,
            lrLearningRate: this.lrRate,
            lrEpochs: this.lrEpochs,
            cnnEpochs: this.cnnEpochs,
            cnnLearningRate: this.cnnRate
        };

        $.ajax({
            url: '/api/update-config',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(conf),
            success: (data) => {
                $(button).parent('#trainParameters').collapse('hide');
                $('#toggle-text').fadeOut(400, function() {
                    $(this).text('Saved').fadeIn(400);
                })
                setTimeout(function(){
                    $('#toggle-text').fadeOut(500, function() {
                        $(this).text('Click to toggle').fadeIn(500);
                    });
                },2000);
            }
        })
        .fail(() => {
            this.clearOutput();
            this.checkConnection();
        });
    }

    checkConnection() {
        const error = "<b>Please make sure the server is running and check its console for further information.</b>";
        $("#error").html(error);
    }
}

$(() => {
    var main = new Main();

    $('#clear').click(() => {
        main.initialize();
    });

    $('#addTrainingData').click(() => {
        main.addTrainingData();
    });

    $('#importFile').change((e) => {
        main.loadImage(e.target.files[0]);
    });

    $('#importFolder').change((e) => {
        main.loadAndUploadImages(e.target);
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

    $('#config-form').submit((e) => {
        main.updateConfig(e.target);
        return false;
    });

});