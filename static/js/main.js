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
        $('#output td, tr').remove();
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
            if (Math.min(...inputs) === 255) {
                return;
            }
            cb(inputs);
        };
        img.src = this.canvas.toDataURL();
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
                        //if (results[classifierIdx] && results[classifierIdx][categoryIdx]) {
                            const result = results[classifierIdx][categoryIdx];
                            cell.text((result*100).toFixed(3)+"%");
                            const mostSuccessfullValue = mostSuccessfullValues[classifierIdx];
                            if (!mostSuccessfullValue || result > mostSuccessfullValue) {
                                mostSuccessfullValues[classifierIdx] = result;
                                mostSuccessfullCells[classifierIdx] = cell;
                            }
                        //}
                    }
                }

                for (let index = 0; index < mostSuccessfullCells.length; index++){
                    mostSuccessfullCells[index].addClass("success");
                }
            }
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
            }
        });
    }

    trainModel(button) {
        $(button).prop('disabled', true)
        button.innerHTML = "Training..."
        $.ajax({
            url: '/api/train-model',
            method: 'POST',
            success: (data) => {
                button.innerHTML = "Train"
                $(button).prop('disabled', false)
            }
        })
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
         main.initialize();
    });

    $('#train').click((e) => {
        main.trainModel(e.target);
    });
});
