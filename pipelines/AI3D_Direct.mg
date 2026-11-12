{
    "header": {
        "releaseVersion": "2025.1.0",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
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
                "comment": "Select a transparent image (or standard photo).",
                "label": "CameraInit",
                "color": ""
            }
        },
        "AI3DGenerator_1": {
            "nodeType": "AI3DGenerator",
            "position": [
                320,
                0
            ],
            "inputs": {
                "input": "{CameraInit_1.output}",
                "modelName": "triposr",
                "executionBackend": "local_wsl_gpu",
                "mcResolution": "192",
                "chunkSize": "4096",
                "lowVramMode": true
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Standalone Generative 3D Mesh (TripoSR) or Gaussians (TripoSplat).",
                "label": "AI3DGenerator",
                "color": "#d62728"
            }
        },
        "Publish_Outputs": {
            "nodeType": "Publish",
            "position": [
                660,
                0
            ],
            "inputs": {
                "inputFiles": [
                    "{AI3DGenerator_1.outputMeshObj}",
                    "{AI3DGenerator_1.outputMeshGlb}",
                    "{AI3DGenerator_1.outputGaussianPly}"
                ],
                "output": "{projectFolder}/Published_AI3D_Direct"
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Exports generated 3D meshes directly.",
                "label": "Publish_Outputs",
                "color": "#2ca02c"
            }
        }
    }
}

