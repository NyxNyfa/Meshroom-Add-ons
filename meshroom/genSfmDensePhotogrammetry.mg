{
    "header": {
        "releaseVersion": "2025.1.0",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "DepthMap": "5.0",
            "DepthMapFilter": "4.0",
            "GenSfM": "1.0",
            "MeshFiltering": "3.0",
            "Meshing": "7.0",
            "PrepareDenseScene": "3.1",
            "Publish": "1.3",
            "Texturing": "6.0"
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
            "inputs": {}
        },
        "GenSfM_1": {
            "nodeType": "GenSfM",
            "position": [
                200,
                0
            ],
            "inputs": {
                "input": "{CameraInit_1.output}",
                "generateSyntheticViews": 24
            }
        },
        "PrepareDenseScene_1": {
            "nodeType": "PrepareDenseScene",
            "position": [
                400,
                0
            ],
            "inputs": {
                "input": "{GenSfM_1.outputSfMDataDense}"
            }
        },
        "DepthMap_1": {
            "nodeType": "DepthMap",
            "position": [
                600,
                0
            ],
            "inputs": {
                "input": "{PrepareDenseScene_1.input}",
                "imagesFolder": "{PrepareDenseScene_1.output}"
            }
        },
        "DepthMapFilter_1": {
            "nodeType": "DepthMapFilter",
            "position": [
                800,
                0
            ],
            "inputs": {
                "input": "{DepthMap_1.input}",
                "depthMapsFolder": "{DepthMap_1.output}"
            }
        },
        "Meshing_1": {
            "nodeType": "Meshing",
            "position": [
                1000,
                0
            ],
            "inputs": {
                "input": "{DepthMapFilter_1.input}",
                "depthMapsFolder": "{DepthMapFilter_1.output}"
            }
        },
        "MeshFiltering_1": {
            "nodeType": "MeshFiltering",
            "position": [
                1200,
                0
            ],
            "inputs": {
                "inputMesh": "{Meshing_1.outputMesh}",
                "keepLargestMeshOnly": false,
                "smoothingIterations": 1
            }
        },
        "Texturing_1": {
            "nodeType": "Texturing",
            "position": [
                1400,
                0
            ],
            "inputs": {
                "input": "{Meshing_1.output}",
                "imagesFolder": "{DepthMap_1.imagesFolder}",
                "inputMesh": "{MeshFiltering_1.outputMesh}",
                "textureSide": 4096,
                "downscale": 1,
                "colorMapping": {
                    "enable": true,
                    "colorMappingFileType": "png"
                },
                "fillHoles": true
            }
        },
        "Publish_1": {
            "nodeType": "Publish",
            "position": [
                1600,
                0
            ],
            "inputs": {
                "inputFiles": [
                    "{Texturing_1.outputMesh}",
                    "{Texturing_1.outputMaterial}",
                    "{Texturing_1.outputTextures}",
                    "{GenSfM_1.outputMeshGlb}",
                    "{GenSfM_1.outputGaussianPly}",
                    "{GenSfM_1.outputSfMDataDense}"
                ]
            }
        }
    }
}
