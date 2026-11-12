{
    "header": {
        "releaseVersion": "2025.1.0",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "RemBG": "1.0",
            "GenSfM": "2.0",
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
                "comment": "Select 2 to 4 uncalibrated photos from random angles.",
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
                "padToSquare": false
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Segments photos into RGBA and masks.",
                "label": "RemBG",
                "color": "#1f77b4"
            }
        },
        "GenSfM_1": {
            "nodeType": "GenSfM",
            "position": [
                580,
                0
            ],
            "inputs": {
                "input": "{RemBG_1.outputSfm}",
                "inputMasks": "{RemBG_1.outputMasks}",
                "maskingStrategy": "external_masks",
                "viewSelectionMode": "all_up_to_4",
                "executionBackend": "local_wsl_gpu",
                "lowVramMode": true
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Generates 3D Textured Mesh and Camera Poses utilizing pre-computed RemBG masks.",
                "label": "GenSfM",
                "color": "#e377c2"
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
                    "{GenSfM_1.outputMeshObj}",
                    "{GenSfM_1.outputSfMData}"
                ],
                "output": "{projectFolder}/Published_GenSfM"
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Exports generated 3D Sparse Point Cloud and Textured Mesh.",
                "label": "Publish_Outputs",
                "color": "#2ca02c"
            }
        }
    }
}

