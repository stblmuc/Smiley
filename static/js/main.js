/* global $ */
class Main {

    constructor() {
        this.canvas = document.getElementById('main');
        this.input = document.getElementById('input');
        this.canvas.width = 449;
        this.canvas.height = 449;
        this.ctx = this.canvas.getContext('2d');
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.image_size = param.image_size;
        this.initialize();
    }

    initialize() {  
        var rect_size = 16 * this.image_size + 1;
        this.canvas.width = rect_size;
        this.canvas.height = rect_size;
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(0, 0, rect_size, rect_size);
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(0, 0, rect_size, rect_size);
        this.ctx.lineWidth = 0.05;
        for (var i = 0; i < this.image_size; i++) {
            this.ctx.beginPath();
            this.ctx.moveTo((i + 1) * 16, 0);
            this.ctx.lineTo((i + 1) * 16, rect_size);
            this.ctx.closePath();
            this.ctx.stroke();

            this.ctx.beginPath();
            this.ctx.moveTo(0, (i + 1) * 16);
            this.ctx.lineTo(rect_size, (i + 1) * 16);
            this.ctx.closePath();
            this.ctx.stroke();
        }
        this.clearOutput();
    }

    onMouseDown(e) {
        this.canvas.style.cursor = 'default';
        this.drawing = true;
        this.prev = this.getPosition(e.clientX, e.clientY);
    }

    onMouseUp() {
        this.drawing = false; // TODO wrong drawInput
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
        this.input.width = 5 * this.image_size;
        this.input.height = 5 * this.image_size;
        var img = new Image();
        img.onload = () => {
            var inputs = [];
            var small = document.createElement('canvas').getContext('2d');
            small.drawImage(img, 0, 0, img.width, img.height, 0, 0, this.image_size, this.image_size);
            var data = small.getImageData(0, 0, this.image_size, this.image_size).data;
            for (var i = 0; i < this.image_size; i++) {
                for (var j = 0; j < this.image_size; j++) {
                    var n = 4 * (i * this.image_size + j);
                    inputs[i * this.image_size + j] = (data[n + 0] + data[n + 1] + data[n + 2]);
                    ctx.fillStyle = 'rgb(' + [data[n + 0], data[n + 1], data[n + 2]].join(',') + ')';
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

    trainModels(button) {
        $(button).prop('disabled', true);
        $(button).text("Training...");
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

    $('#trainModels').click((e) => {
        main.trainModels(e.target);
    });

    $('#deleteModels').click((e) => {
        main.deleteAllModels(e.target);
    });
});