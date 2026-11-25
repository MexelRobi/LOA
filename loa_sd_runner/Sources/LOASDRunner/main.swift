import CoreML
import Foundation
import ImageIO
import StableDiffusion
import UniformTypeIdentifiers

enum RunnerError: Error {
    case invalidArguments
    case unsupportedSystem
    case noImage
    case cannotCreateOutput
    case cannotSaveOutput
}

func writePNG(_ image: CGImage, to output: URL) throws {
    let data = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(
        data,
        UTType.png.identifier as CFString,
        1,
        nil
    ) else {
        throw RunnerError.cannotCreateOutput
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
        throw RunnerError.cannotSaveOutput
    }
    try (data as Data).write(to: output, options: .atomic)
}

@available(macOS 13.1, *)
func generateImage() throws {
    guard (5...6).contains(CommandLine.arguments.count),
          let stepCount = Int(CommandLine.arguments[4]),
          (1...100).contains(stepCount) else {
        throw RunnerError.invalidArguments
    }

    let resources = URL(fileURLWithPath: CommandLine.arguments[1])
    let prompt = CommandLine.arguments[2]
    let output = URL(fileURLWithPath: CommandLine.arguments[3])
    let progressFile = CommandLine.arguments.count == 6
        ? URL(fileURLWithPath: CommandLine.arguments[5])
        : nil

    let modelConfiguration = MLModelConfiguration()
    modelConfiguration.computeUnits = .cpuAndNeuralEngine
    let pipeline = try StableDiffusionPipeline(
        resourcesAt: resources,
        controlNet: [],
        configuration: modelConfiguration,
        reduceMemory: true
    )
    try pipeline.loadResources()

    var configuration = StableDiffusionPipeline.Configuration(prompt: prompt)
    configuration.stepCount = stepCount
    configuration.seed = UInt32.random(in: 0...UInt32.max)
    configuration.schedulerType = .dpmSolverMultistepScheduler
    var progressWriteError: Error?
    let images = try pipeline.generateImages(configuration: configuration) { progress in
        guard let progressFile else {
            return true
        }
        guard let image = progress.currentImages.first ?? nil else {
            return true
        }
        do {
            try writePNG(image, to: output)
            try "\(progress.step + 1)/\(progress.stepCount)"
                .write(to: progressFile, atomically: true, encoding: .utf8)
            return true
        } catch {
            progressWriteError = error
            return false
        }
    }
    if let progressWriteError {
        throw progressWriteError
    }
    guard !images.isEmpty, let image = images[0] else {
        throw RunnerError.noImage
    }
    try writePNG(image, to: output)
    pipeline.unloadResources()
}

if #available(macOS 13.1, *) {
    do {
        try generateImage()
    } catch {
        fputs("loa-sd-runner: \(error)\n", stderr)
        exit(EXIT_FAILURE)
    }
} else {
    fputs("loa-sd-runner requires macOS 13.1 or later.\n", stderr)
    exit(EXIT_FAILURE)
}
