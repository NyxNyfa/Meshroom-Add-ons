{
    "header": {
        "releaseVersion": "2025.1.0",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "RemBG": "1.0",
            "AI3DGenerator": "1.0",
            "MeshFiltering": "3.0",
            "MeshDecimate": "1.0",
            "Publish": "1.3"
        },
        "template": true
    },
    "graph": {
        "CameraInit_1": {
            "nodeType": "CameraInit",
            "position": [
                0,
                0
            ],
            "inputs": {
                "allowSingleView": 1
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Select an object photo.",
                "label": "CameraInit",
                "color": ""
            }
        },
        "RemBG_1": {
            "nodeType": "RemBG",
            "position": [
                280,
                0
            ],
            "inputs": {
                "input": "{CameraInit_1.output}",
                "model": "u2net",
                "executionBackend": "local_wsl_gpu",
                "alphaMatting": false,
                "padToSquare": true
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Pre-processes single image with AI background removal and 1:1 padding.",
                "label": "RemBG",
                "color": "#1f77b4"
            }
        },
        "AI3DGenerator_1": {
            "nodeType": "AI3DGenerator",
            "position": [
                580,
                0
            ],
            "inputs": {
                "input": "{RemBG_1.outputSfm}",
                "modelName": "triposr",
                "executionBackend": "local_wsl_gpu",
                "mcResolution": "192",
                "chunkSize": "4096",
                "numGaussians": "131072",
                "lowVramMode": true
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Feed-forward generative AI reconstructing 3D mesh (TripoSR) or Gaussians (TripoSplat).",
                "label": "AI3DGenerator",
                "color": "#d62728"
            }
        },
        "MeshFiltering_1": {
            "nodeType": "MeshFiltering",
            "position": [
                880,
                0
            ],
            "inputs": {
                "inputMesh": "{AI3DGenerator_1.outputMeshObj}",
                "keepLargestMeshOnly": true,
                "smoothingIterations": 5,
                "lambda": 0.5
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Removes disconnected floating geometry and smooths marching cubes surface.",
                "label": "MeshFiltering",
                "color": "#ff7f0e"
            }
        },
        "MeshDecimate_1": {
            "nodeType": "MeshDecimate",
            "position": [
                1180,
                0
            ],
            "inputs": {
                "input": "{MeshFiltering_1.outputMesh}",
                "simplificationFactor": 0.5
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Reduces triangle count for real-time game engines and 3D applications.",
                "label": "MeshDecimate",
                "color": "#bcbd22"
            }
        },
        "Publish_Outputs": {
            "nodeType": "Publish",
            "position": [
                1480,
                0
            ],
            "inputs": {
                "inputFiles": [
                    "{MeshDecimate_1.outputMesh}",
                    "{AI3DGenerator_1.outputMeshGlb}",
                    "{AI3DGenerator_1.outputGaussianPly}"
                ],
                "output": "{projectFolder}/Published_AI3D"
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Exports final polished 3D assets.",
                "label": "Publish_Outputs",
                "color": "#2ca02c"
            }
        }
    }
}

