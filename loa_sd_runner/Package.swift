// swift-tools-version: 5.8
import PackageDescription

let package = Package(
    name: "loa-sd-runner",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "loa-sd-runner", targets: ["LOASDRunner"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/apple/ml-stable-diffusion.git",
            exact: "1.1.1"
        ),
    ],
    targets: [
        .executableTarget(
            name: "LOASDRunner",
            dependencies: [
                .product(name: "StableDiffusion", package: "ml-stable-diffusion"),
            ]
        ),
    ]
)
