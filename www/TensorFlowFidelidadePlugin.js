var exec = require('cordova/exec');
/**
 * Returns tensorflow value
 * 
 * @param modelName the model name of the file already exists
 * @param imagePath the image path with image content
 * @param successCallback the success
 * @param errorCallback the error
 */
exports.loadModel = function(modelName, imagePath, successCallback, errorCallback) {
    exec(successCallback, errorCallback,  'TensorFlowFidelidadePlugin', 'loadModel', [modelName, imagePath]);
};