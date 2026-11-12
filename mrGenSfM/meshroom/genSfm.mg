{
    "header": {
        "releaseVersion": "2025.1.0",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "GenSfM": "1.0",
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
            "inputs": {},
            "internalInputs": {
                "invalidation": "",
                "comment": "Select 2 to 4 uncalibrated object photos from random viewpoints.",
                "label": "CameraInit",
                "color": ""
            }
        },
        "GenSfM_1": {
            "nodeType": "GenSfM",
            "position": [
                280,
                0
            ],
            "inputs": {
                "input": "{CameraInit_1.output}",
                "maskingStrategy": "auto_rembg",
                "executionBackend": "local_wsl_gpu",
                "lowVramMode": true
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "AI Core Reconstructor: Generates complete 360-degree textured mesh, Gaussian splats, and camera poses with built-in AI background removal.",
                "label": "GenSfM",
                "color": "#2b7bba"
            }
        },
        "Publish_Outputs": {
            "nodeType": "Publish",
            "position": [
                620,
                0
            ],
            "inputs": {
                "inputFiles": [
                    "{GenSfM_1.outputMeshObj}",
                    "{GenSfM_1.outputMeshMtl}",
                    "{GenSfM_1.outputMeshTexture}",
                    "{GenSfM_1.outputMeshGlb}",
                    "{GenSfM_1.outputGaussianPly}",
                    "{GenSfM_1.outputSfMData}"
                ],
                "output": "{projectFolder}/Published_Output"
            },
            "internalInputs": {
                "invalidation": "",
                "comment": "Publisher: Exports 360-degree Textured Mesh (OBJ+MTL+PNG), Binary GLB, 3D Gaussian Splats (PLY), and SfMData.",
                "label": "Publish_Outputs",
                "color": "#9467bd"
            }
        }
    }
}
