import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

/// Image picker for selecting images or folders
struct ImagePickerView: View {
    @Environment(\.dismiss) private var dismiss
    let onImageSelected: (URL) -> Void
    let onImagesSelected: ([URL]) -> Void
    let onFolderSelected: (URL) -> Void
    let onProjectSelected: (URL) -> Void
    var onCloudflareSettingsTapped: (() -> Void)? = nil
    var onHILDashboardTapped: (() -> Void)? = nil
    var isHILConfigured: Bool = false

    @State private var selectedItems: [PhotosPickerItem] = []
    @State private var showingImagePicker = false
    @State private var showingFolderPicker = false
    @State private var showingProjectPicker = false

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .foregroundColor(.blue)

                Spacer()

                Text("Import")
                    .font(.headline)
                    .foregroundColor(.white)

                Spacer()

                Button("Cancel") { }
                    .opacity(0)
            }
            .padding()
            .background(Color(white: 0.15))

            // Content
            VStack(spacing: 20) {
                Spacer().frame(height: 20)

                // Photos Library (multi-select)
                PhotosPicker(selection: $selectedItems, matching: .images) {
                    HStack {
                        Image(systemName: "photo.on.rectangle")
                            .font(.title2)
                        Text("From Photos Library")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                .onChange(of: selectedItems) { _, newItems in
                    Task { await loadImages(from: newItems) }
                }

                // Single image
                Button(action: { showingImagePicker = true }) {
                    HStack {
                        Image(systemName: "doc.badge.plus")
                            .font(.title2)
                        Text("Single Image from Files")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.orange)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }

                // Folder
                Button(action: { showingFolderPicker = true }) {
                    HStack {
                        Image(systemName: "folder.badge.plus")
                            .font(.title2)
                        Text("Folder (Multiple Images)")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }

                Divider()
                    .background(Color.gray)
                    .padding(.vertical, 10)

                // Open Project
                Button(action: { showingProjectPicker = true }) {
                    HStack {
                        Image(systemName: "folder.badge.gearshape")
                            .font(.title2)
                        Text("Open Project Folder")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.purple)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }

                // Cloudflare / HIL section
                Divider()
                    .background(Color.gray)
                    .padding(.vertical, 10)

                Button(action: {
                    dismiss()
                    onCloudflareSettingsTapped?()
                }) {
                    HStack {
                        Image(systemName: "cloud")
                            .font(.title2)
                        Text("Cloudflare Settings")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.cyan.opacity(0.8))
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }

                if isHILConfigured {
                    Button(action: {
                        dismiss()
                        onHILDashboardTapped?()
                    }) {
                        HStack {
                            Image(systemName: "list.bullet.rectangle")
                                .font(.title2)
                            Text("HIL Dashboard")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.orange.opacity(0.8))
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    }
                }

                Spacer()

                Text("Import: copy images to current project\nOpen Project: switch to different project")
                    .font(.caption)
                    .foregroundColor(.gray)
                    .padding(.bottom, 20)
            }
            .padding(.horizontal, 24)
        }
        .background(Color(white: 0.1))
        .sheet(isPresented: $showingImagePicker) {
            DocumentPickerView(contentTypes: [.image]) { url in
                handleImageImport(url)
            }
        }
        .sheet(isPresented: $showingFolderPicker) {
            DocumentPickerView(contentTypes: [.folder]) { url in
                onFolderSelected(url)
                dismiss()
            }
        }
        .sheet(isPresented: $showingProjectPicker) {
            DocumentPickerView(contentTypes: [.folder]) { url in
                onProjectSelected(url)
                dismiss()
            }
        }
    }

    private func loadImages(from items: [PhotosPickerItem]) async {
        guard !items.isEmpty else { return }

        var urls: [URL] = []
        for item in items {
            do {
                if let data = try await item.loadTransferable(type: Data.self) {
                    let tempURL = FileManager.default.temporaryDirectory
                        .appendingPathComponent(UUID().uuidString)
                        .appendingPathExtension("png")
                    try data.write(to: tempURL)
                    urls.append(tempURL)
                }
            } catch {
                print("[PhotosPicker] Failed to load item: \(error)")
            }
        }

        guard !urls.isEmpty else { return }

        await MainActor.run {
            if urls.count == 1 {
                onImageSelected(urls[0])
            } else {
                onImagesSelected(urls)
            }
            dismiss()
        }
    }

    private func handleImageImport(_ url: URL) {
        let didStartAccessing = url.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing { url.stopAccessingSecurityScopedResource() }
        }

        do {
            let data = try Data(contentsOf: url)
            let tempURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(url.lastPathComponent)
            try data.write(to: tempURL)
            onImageSelected(tempURL)
            dismiss()
        } catch {
            print("[ImageImport] Failed: \(error)")
        }
    }
}

// MARK: - Document Picker

struct DocumentPickerView: UIViewControllerRepresentable {
    let contentTypes: [UTType]
    let onPicked: (URL) -> Void

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: contentTypes)
        picker.delegate = context.coordinator
        picker.allowsMultipleSelection = false
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onPicked: onPicked)
    }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPicked: (URL) -> Void

        init(onPicked: @escaping (URL) -> Void) {
            self.onPicked = onPicked
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }
            onPicked(url)
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {}
    }
}

// MARK: - Preview

#Preview {
    ImagePickerView(
        onImageSelected: { url in print("Image: \(url)") },
        onImagesSelected: { urls in print("Images: \(urls.count)") },
        onFolderSelected: { url in print("Folder: \(url)") },
        onProjectSelected: { url in print("Project: \(url)") }
    )
}
