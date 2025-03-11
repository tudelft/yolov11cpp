package com.example.opencv_tutorial

import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

/**
 * Diagnostic activity for detailed model inspection
 * This helps identify issues with model loading on physical devices
 */
class ModelParseActivity : AppCompatActivity() {
    private lateinit var resultText: TextView
    private val scope = CoroutineScope(Dispatchers.Main)
    
    companion object {
        private const val TAG = "ModelParse"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_parse)
        
        resultText = findViewById(R.id.modelParseResultText)
        resultText.text = "Analyzing TFLite model..."
        
        // Run model inspection in background
        scope.launch {
            try {
                val results = withContext(Dispatchers.IO) {
                    analyzeModels()
                }
                resultText.text = results
            } catch (e: Exception) {
                Log.e(TAG, "Error during model analysis", e)
                resultText.text = "Error analyzing models:\n${e.message}\n\n${e.stackTraceToString()}"
            }
        }
    }
    
    private fun analyzeModels(): String {
        val result = StringBuilder()
        result.append("TFLite Model Analysis\n")
        result.append("====================\n\n")
        
        val modelFiles = listOf(
            "best_float16.tflite",
            "best_float32.tflite", 
            "best.tflite"
        )
        
        for (modelFile in modelFiles) {
            try {
                result.append("Model: $modelFile\n")
                result.append("-----------------\n")
                
                // Check if file exists
                try {
                    assets.open(modelFile).close()
                    result.append("File exists in assets: Yes\n")
                } catch (e: Exception) {
                    result.append("File exists in assets: No\n")
                    result.append("\n")
                    continue
                }
                
                // Extract model to temp file for analysis
                val tempFile = extractModelToTemp(modelFile)
                
                result.append("File size: ${tempFile.length()} bytes\n")
                
                // Basic header verification
                val isValidFlatBuffer = checkFlatBufferHeader(tempFile)
                result.append("Valid FlatBuffer header: $isValidFlatBuffer\n")
                
                // Try to parse model metadata
                try {
                    val metadata = parseModelMetadata(tempFile)
                    result.append(metadata)
                } catch (e: Exception) {
                    result.append("Metadata extraction failed: ${e.message}\n")
                }
                
                // Try basic TFLite interpreter creation
                try {
                    testInterpreterCreation(modelFile)
                    result.append("Interpreter creation: Success\n")
                } catch (e: Exception) {
                    result.append("Interpreter creation failed: ${e.message}\n")
                }
                
                result.append("\n")
                
            } catch (e: Exception) {
                result.append("Error analyzing $modelFile: ${e.message}\n\n")
            }
        }
        
        // Add device information
        result.append("Device Information\n")
        result.append("-----------------\n")
        result.append("Manufacturer: ${android.os.Build.MANUFACTURER}\n")
        result.append("Model: ${android.os.Build.MODEL}\n")
        result.append("Android version: ${android.os.Build.VERSION.RELEASE} (SDK ${android.os.Build.VERSION.SDK_INT})\n")
        result.append("ABI: ${android.os.Build.SUPPORTED_ABIS.joinToString()}\n")
        
        return result.toString()
    }
    
    private fun extractModelToTemp(modelFile: String): File {
        val file = File(cacheDir, "temp_$modelFile")
        
        assets.open(modelFile).use { input ->
            FileOutputStream(file).use { output ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (input.read(buffer).also { read = it } != -1) {
                    output.write(buffer, 0, read)
                }
                output.flush()
            }
        }
        
        return file
    }
    
    private fun checkFlatBufferHeader(file: File): Boolean {
        return file.inputStream().use { input ->
            val header = ByteArray(8)
            val bytesRead = input.read(header)
            
            // Check standard FlatBuffer header
            (bytesRead == 8) &&
                   header[0].toInt() == 0x18 && 
                   header[1].toInt() == 0x00 && 
                   header[2].toInt() == 0x00 && 
                   header[3].toInt() == 0x00
        }
    }
    
    private fun parseModelMetadata(file: File): String {
        val result = StringBuilder()
        
        try {
            val mappedBuffer = file.inputStream().channel.map(
                FileChannel.MapMode.READ_ONLY, 0, file.length()
            )
            
            val metadataExtractor = MetadataExtractor(mappedBuffer)
            
            // Check if model has metadata
            if (metadataExtractor.hasMetadata()) {
                result.append("Has metadata: Yes\n")
                
                // Get model description
                val modelMetadata = metadataExtractor.modelMetadata
                if (modelMetadata != null) {
                    result.append("Model name: ${modelMetadata.name()}\n")
                    result.append("Model description: ${modelMetadata.description()}\n")
                    result.append("Model version: ${modelMetadata.version()}\n")
                }
                
                // Get input/output tensors
                val inputTensorCount = metadataExtractor.inputTensorCount
                val outputTensorCount = metadataExtractor.outputTensorCount
                
                result.append("Input tensors: $inputTensorCount\n")
                result.append("Output tensors: $outputTensorCount\n")
                
                for (i in 0 until inputTensorCount) {
                    val tensorMetadata = metadataExtractor.getInputTensorMetadata(i)
                    result.append("Input #$i: ${tensorMetadata.name()}, ")
                    result.append("type: ${tensorMetadata.tensorType().name}\n")
                }
            } else {
                result.append("Has metadata: No\n")
            }
            
            // Get basic model info directly from the buffer
            try {
                mappedBuffer.rewind()
                val model = org.tensorflow.lite.schema.Model.getRootAsModel(mappedBuffer)
                result.append("Model version: ${model.version()}\n")
                result.append("Operator codes: ${model.operatorCodesLength()}\n")
                result.append("Subgraphs: ${model.subgraphsLength()}\n")
                
                if (model.subgraphsLength() > 0) {
                    val subgraph = model.subgraphs(0)
                    if (subgraph != null) {
                        result.append("Inputs: ${subgraph.inputsLength()}, ")
                        result.append("Outputs: ${subgraph.outputsLength()}\n")
                    }
                }
            } catch (e: Exception) {
                result.append("Schema parse error: ${e.message}\n")
            }
            
        } catch (e: Exception) {
            result.append("Metadata extraction error: ${e.message}\n")
        }
        
        return result.toString()
    }
    
    private fun testInterpreterCreation(modelFile: String) {
        val assetFd = assets.openFd(modelFile)
        val fileChannel = FileInputStream(assetFd.fileDescriptor).channel
        val mappedBuffer = fileChannel.map(
            FileChannel.MapMode.READ_ONLY, 
            assetFd.startOffset, 
            assetFd.declaredLength
        )
        
        // Test creating interpreter with basic options
        val options = org.tensorflow.lite.Interpreter.Options()
        val interpreter = org.tensorflow.lite.Interpreter(mappedBuffer, options)
        
        // Log the model info
        val inputs = interpreter.inputTensorCount
        val outputs = interpreter.outputTensorCount
        Log.d(TAG, "Model has $inputs inputs and $outputs outputs")
        
        // Clean up
        interpreter.close()
        fileChannel.close()
        assetFd.close()
    }
}
