{
    "header": {
        "releaseVersion": "2025.1.0",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "RemBG": "1.0",
            "AI3DGenerator": "1.0",
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
                "comment": "Select a single object photo.",
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
        "Publish_Outputs": {
            "nodeType": "Publish",
            "position": [
                880,
                0
            ],
            "inputs": {
                "inputFiles": [
                    "{AI3DGenerator_1.outputMeshObj}",
                    "{AI3DGenerator_1.outputMeshGlb}",
                    "{AI3DGenerator_1.outputGaussianPly}"
                ],
                "output": "{projectFolder}/Published_AIGen3D"
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Exports generated 3D assets (OBJ, GLB PBR, and Gaussian PLY).",
                "label": "Publish_Outputs",
                "color": "#2ca02c"
            }
        }
    }
}

