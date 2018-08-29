/* global $ */
class Main {
    constructor() {
        this.canvas = $('#main')[0];
        this.ctx = this.canvas.getContext('2d');
        this.input = $('#input')[0];  

        this.numAugm = param.numAugm;
        this.batchSize = param.batchSize;
        this.srRate = param.srRate;
        this.srEpochs = param.srEpochs;
        this.cnnRate = param.cnnRate;
        this.cnnEpochs = param.cnnEpochs;
        this.image_size = param.image_size;
        this.rect_size = 448; // 16 * 28
        this.col_width = this.rect_size / this.image_size; // for the grid

        this.canvas.width = this.rect_size;
        this.canvas.height = this.rect_size;
        this.input.width = 5 * this.image_size;
        this.input.height = 5 * this.image_size;

        var catsList = $('#trainingDataLabelOptions')[0];
        this.cats = param.categories;
        this.cats.forEach(function(item){
            var option = document.createElement('option');
            $(option).val(item);
            catsList.append(option);
        });

        this.makeDrawActive();

        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));

        this.createUserCategoryButtons();
        this.addNumberToCategories();
        this.initializeConfigValues();
        this.initialize();
    }

    initialize() {
        this.clearOutput();

        if (!!this.video) {
            this.video.play();
        } else {
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
        }
    }

    initializeConfigValues() {
        $('#num-augm').val(this.numAugm);
        $('#batch-size').val(this.batchSize);
        $('#sr-rate').val(this.srRate);
        $('#sr-epochs').val(this.srEpochs);
        $('#cnn-rate').val(this.cnnRate);
        $('#cnn-epochs').val(this.cnnEpochs);
    }

    addNumberToCategories() {
        for (var key in param.cat_number) {
            // check if the property/key is defined in the object itself, not in parent
            if (param.cat_number.hasOwnProperty(key)) {
                this.addNewNumberToCategory(key, param.cat_number[key]);
            }
        }
    }

    addNewNumberToCategory(category, number) {
        var numberDiv = document.createElement('div');
        $(numberDiv).html(" (" + number + ")").addClass("inline");
        numberDiv.id = category + "-number";

        var x = $('.add-emoji-data').filter(function(){return this.value==category})[0];
        x.insertBefore(numberDiv, x.childNodes[1]);
    }

    createUserCategoryButtons() {
        param.user_categories.forEach((item) => {
            this.addUserCategoryButton(item);
        });
    }

    addUserCategoryButton(label) {
        var catsButtons = $('#ownCategories')[0];
        var outerDiv = document.createElement('div');
        $(outerDiv).addClass("btn btn-outline-secondary add-emoji-data")
        .html(label).val(label).click((e) => {
            this.addTrainingData(e.currentTarget, $(e.currentTarget).val());
        });
        var newButton = document.createElement('div');
        $(newButton).addClass("cross-img").html("&#10060;").click((e) => {
            outerDiv.remove();
            this.deleteCategory(label);
            e.stopPropagation();
        }).appendTo(outerDiv);
        catsButtons.append(outerDiv);
    }

    deleteCategory(label) {
        $("#trainingDataLabelOptions option[value='"+label+"']").remove(); // delete cat from datalist options
        var index = this.cats.indexOf(label);
        if (index !== -1) {
            this.cats.splice(index, 1);
        }

        const catData = {
            cat: label
        };
        $.ajax({
            url: '/api/delete-category',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(catData),
            success: (data) => {
                this.initialize();
            }
        })
        .fail(() => {
            this.clearOutput();
            this.checkConnection();
        });
    }

    onMouseDown(e) {
        if (!!this.video) return; // don't draw in camera mode

        this.canvas.style.cursor = 'default';
        this.drawing = true;
        this.prev = this.getPosition(e.clientX, e.clientY);
    }

    onMouseUp() {
        if (!!this.video) {
            this.video.pause();
            this.recogniseInput((input) => {})
        }

        this.drawing = false;
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
            var input = [];
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

                    input[i * this.image_size + j] = grayscale;
                    ctx.fillStyle = 'rgb(' + Array(3).fill(grayscale) + ')';
                    ctx.fillRect(j * 5, i * 5, 5, 5);
                }
            }
            if (Math.min(...input) === 255) {
                return;
            }
            cb(input);
        };
        img.src = this.canvas.toDataURL();
    }

    clearOutput() {
        $("#error").text("");
        $('#output td, #output tr').remove();
    }

    recogniseInput(cb) {
        if (!!this.video) this.video.pause();

        this.drawInput((input) => {
            (typeof cb == 'function') ? cb(input) : this.loadOutput(input);
        });
    }

    loadOutput(input) {
        $.ajax({
            url: '/api/smiley',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(input),
            success: (data) => {
                const categories = data.categories;
                const error = data.error;
                var classifiers = data.classifiers;
                var results = data.results;

                if (error) {
                    $("#error").html(error);
                } else {
                    $("#error").text("");
                }

                // Don't display table if results contain empty arrays
                if (!results.filter((e)=>{return e.length}).length)
                    return;
                else {
                    // concat average to results
                    const average = arr => arr[0].map((v,i) => (v+arr[1][i]) / 2);
                    classifiers = classifiers.concat(["Average"]);
                    results = results.concat([average(results)]);
                }

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
                    const textElement = $("<span class='button-own-image "+categories[categoryIdx]+"-img'>");
                    textElement.text(categories[categoryIdx]);
                    categoryNameCell.append(textElement);
                    row.append(categoryNameCell);
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

    addTrainingData(button, label) {
        if (label) {
            this.recogniseInput((input) => {
                const uploadData = {
                    cat: label,
                    img: input
                };

                $(button).fadeOut(400).fadeIn(400);
                var blink = setInterval(function(){
                    $(button).fadeOut(400).fadeIn(400);
                }, 1000);

                this.uploadTrainingData(uploadData, blink);
            });
        } else {
            alert("Please assign a category for the data");
        }
    }

    uploadTrainingData(input, blink) {
        $.ajax({
            url: '/api/generate-training-example',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(input),
            success: (data) => {
                this.initialize();
                const error = data.error;
                if (error) {
                    $("#error").html(error);
                }

                var label = input.cat;
                if (!this.cats.includes(label)) {
                    this.cats.push(label)
                    var catsList = $('#trainingDataLabelOptions')[0];
                    var option = document.createElement('option');
                    $(option).val(label);
                    catsList.append(option);

                    this.addUserCategoryButton(label);
                }

                var catNumber = $('#'+label+'-number');
                if(catNumber.length > 0) {
                    var temp = catNumber.html();
                    var n = parseInt(temp.substring(2, temp.length-1));
                    catNumber.html(" (" + (n + 1) + ")");
                } else {
                    this.addNewNumberToCategory(label, 1 );
                }
            }
        })
        .always(() => {
            clearInterval(blink);
        })
        .fail(() => {
            this.clearOutput();
            this.checkConnection();
        });
    }

    useModeDraw(button) {
        if (!!this.video) {
            this.video.pause();
            this.video = null;
        }

        this.initialize();
    }

    useModeCamera(button) {
        if (!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
            if (!this.video) {
                var constraints = {video: {width: this.rect_size, height: this.rect_size, facingMode: "user", frameRate: 10}};
                
                navigator.mediaDevices.enumerateDevices()
                .then((deviceInfos) => {
                    /* sets video_device_id to the last webcam found */
                    for (var i = 0; i < deviceInfos.length; ++i) 
                        if (deviceInfos[i].kind === 'videoinput')
                            this.video_device_id = deviceInfos[i].deviceId;

                    constraints['video']['deviceId'] = {exact: this.video_device_id};
                    return navigator.mediaDevices.getUserMedia(constraints);
                })
                .then((mediaStream) => {
                    const ctx = this.ctx;
                    const rect_size = this.rect_size;

                    this.video = document.createElement('video');
                    this.video.srcObject = mediaStream;
                    this.video.addEventListener('play', function(){
                        var $this = this;
                        (function loop() {
                            if (!$this.paused && !$this.ended) {
                                ctx.drawImage($this, 0, 0, rect_size, rect_size);
                                setTimeout(loop, 1000 / constraints['video']['frameRate']); // drawing at 10fps
                            }
                        })();
                    }, 0);
                    this.video.play();
                })
                .catch(this.makeDrawActive);
            } else {
                this.initialize();
                this.makeDrawActive();
            }
        } else {
            alert('getUserMedia() is not supported by your browser');
            this.makeDrawActive();
        }
    }

    makeDrawActive() {
        $('#modeDraw').addClass("menu-active");
        $('#modeCamera').removeClass("menu-active");
        $('#takePhoto').hide();
    }

    trainModels(button) {
        $(button).prop('disabled', true)
        .addClass('progress-bar-striped progress-bar-animated');

        var update_progress = setInterval(function() {
            $.ajax({
                url: '/api/train-progress',
                success: (data) => {
                    $(button).css('width', data.progress + '%')
                }
            })
        }, 800);

        $.ajax({
            url: '/api/train-models',
            method: 'POST',
            success: (data) => {
                const error = data.error;
                if (error) {
                    $("#error").html(error);
                } else {
                    $("#error").text("");

                    this.recogniseInput();
                }

            }
        })
        .always(() => {
            clearInterval(update_progress);
            $(button).css('width', '100%')
            .removeClass('progress-bar-striped progress-bar-animated')
            .prop('disabled', false);
        })
        .fail(() => {
            this.clearOutput();
            this.checkConnection();
        });
    }

    updateConfig() {
        this.numAugm = $('#num-augm').val();
        this.batchSize = $('#batch-size').val();
        this.srRate = $('#sr-rate').val();
        this.srEpochs = $('#sr-epochs').val();
        this.cnnRate = $('#cnn-rate').val();
        this.cnnEpochs = $('#cnn-epochs').val();

        const conf = {
            numberAugmentations: this.numAugm,
            batchSize: this.batchSize,
            srLearningRate: this.srRate,
            srEpochs: this.srEpochs,
            cnnEpochs: this.cnnEpochs,
            cnnLearningRate: this.cnnRate
        };

        $.ajax({
            url: '/api/update-config',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(conf),
            success: (data) => {
                // $('#trainParameters').collapse('hide');
                $('#trainParameters input').removeClass('updating');
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


    // loadImage(data, cb) {
    //     var img = new Image();
    //     img.onload = () => {
    //         this.initialize();
    //         var imgSize = Math.min(img.width, img.height);
    //      var left = (img.width - imgSize) / 2;
    //      var top = (img.height - imgSize) / 2;

    //         // draw squared-up image in canvas
    //         this.ctx.drawImage(img, left, top, imgSize, imgSize, 0, 0, this.ctx.canvas.width, this.ctx.canvas.height);

    //         this.drawInput((input) => {
    //             if (typeof cb == 'function')
    //                 cb(data, input);
    //             else
    //                 this.loadOutput(input);
    //         });
    //     }
    //     img.src = window.URL.createObjectURL(data)
    // }

    // loadAndUploadImages(target) {
    //     function cb(data, input) {
    //         var path = data.webkitRelativePath.split("/");
    //         var label = path[path.length - 2];
    //         if (label) {
    //             const uploadData = {
    //                 cat: label,
    //                 img: input
    //             };
    //             $.ajax({
    //                 url: '/api/generate-training-example',
    //                 method: 'POST',
    //                 contentType: 'application/json',
    //                 data: JSON.stringify(uploadData),
    //                 success: (data) => {
    //                 }
    //             })
    //         } else {
    //             alert("Please select a folder of one category or of one image size");
    //         };
    //     }

    //     for (var i = 0; i < target.files.length; i++) {
    //         this.loadImage(target.files[i], cb);
    //     }
    // }

    // getConsoleOutput(firstCall) {
    //     var obj = $('#consoleOutput .card-body');
    //     $.ajax({
    //         url: '/api/get-console-output',
    //         success: (data) => {
    //             if (!!firstCall) obj.append("done!<br>");
    //             if (data.out) obj.append(data.out.replace(/(\r\n|\n|\r)/gm, "<br>"));
    //         }
    //     })
    //     .fail(() => {
    //         obj.html("Connection failed.<br>");
    //     });
    // }
}

$(() => {
    var main = new Main();

    $('#modeDraw').click((e) => {
        main.useModeDraw(e.currentTarget);
        main.makeDrawActive();
    });

    $('#modeCamera').click((e) => {
        main.useModeCamera(e.currentTarget);
        $(e.currentTarget).addClass("menu-active");
        $('#modeDraw').removeClass("menu-active");
        $('#takePhoto').show();
    });

    $('#takePhoto').click((e) => {
        main.onMouseUp();
    });

    $('#clear').click(() => {
        main.initialize();
    });

    $('#recognise').click(() => {
        main.recogniseInput();
    });

    $('#addTrainingData').click((e) => {
        main.addTrainingData(e.currentTarget, $('#trainingDataLabel').val());
    });

    $('.button-own-image').click((e) => {
        main.addTrainingData(e.currentTarget, $(e.currentTarget).val());
    });

    $('#trainModels').click((e) => {
        main.trainModels(e.currentTarget);
    });

    $('#config-form').submit((e) => {
        main.updateConfig();
        return false;
    });

    $('#config-form input').each(function() {
        $(this).change((e) => {
            $(this).addClass('updating');
            this.timeout = setTimeout(() => {
                $('#config-form').submit();
            }, 2000);
        })
    })

    /*$('#importFile').change((e) => {
        main.loadImage(e.currentTarget.files[0]);
    });

    $('#importFolder').change((e) => {
        main.loadAndUploadImages(e.currentTarget);
    });

    $('#deleteModels').click((e) => {
        main.deleteAllModels(e.currentTarget);
    });*/
});