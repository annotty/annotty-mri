import SwiftUI

/// Dashboard showing HIL server images with batch selection and import
struct HILDashboardView: View {
    @ObservedObject var hilViewModel: HILViewModel
    @ObservedObject var settings: HILSettings
    var localImageNames: Set<String> = []
    let onImportSelected: (Set<String>) -> Void
    @Environment(\.dismiss) private var dismiss

    @State private var selectedImageIds: Set<String> = []

    private var allSelected: Bool {
        !hilViewModel.imageList.isEmpty && selectedImageIds.count == hilViewModel.imageList.count
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Status bar
                statusBar
                    .padding()

                Divider()

                // Select All bar
                if !hilViewModel.imageList.isEmpty {
                    selectAllBar
                        .padding(.horizontal)
                        .padding(.vertical, 8)

                    Divider()
                }

                if hilViewModel.isLoading {
                    Spacer()
                    ProgressView("Loading images...")
                    Spacer()
                } else if hilViewModel.imageList.isEmpty {
                    Spacer()
                    Text("No images on server")
                        .foregroundColor(.secondary)
                    Spacer()
                } else {
                    // Image list
                    ScrollView {
                        LazyVStack(spacing: 8) {
                            ForEach(hilViewModel.imageList) { image in
                                imageRow(image)
                            }
                        }
                        .padding()
                    }
                }

                Divider()

                // Bottom controls
                bottomControls
                    .padding()
            }
            .navigationTitle("HIL Dashboard")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .task {
                await hilViewModel.connect()
            }
        }
    }

    // MARK: - Status Bar

    private var statusBar: some View {
        HStack {
            // Connection status
            HStack(spacing: 6) {
                Circle()
                    .fill(hilViewModel.isConnected ? .green : .red)
                    .frame(width: 10, height: 10)
                Text(hilViewModel.isConnected ? "Connected" : "Disconnected")
                    .font(.caption)
            }

            Spacer()

            // Progress
            if let info = hilViewModel.serverInfo {
                VStack(alignment: .trailing, spacing: 2) {
                    Text("\(info.labeledImages) / \(info.totalImages) labeled")
                        .font(.caption)
                    ProgressView(value: Double(info.labeledImages), total: max(Double(info.totalImages), 1))
                        .frame(width: 120)
                }
            }
        }
    }

    // MARK: - Select All Bar

    private var selectAllBar: some View {
        HStack {
            Spacer()
            Button {
                if allSelected {
                    selectedImageIds.removeAll()
                } else {
                    selectedImageIds = Set(hilViewModel.imageList.map(\.id))
                }
            } label: {
                HStack(spacing: 6) {
                    Text("Select All")
                        .font(.subheadline)
                        .foregroundColor(.primary)
                    checkboxIcon(checked: allSelected)
                }
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - Image Row

    private func imageRow(_ image: HILServerClient.ImageInfo) -> some View {
        let isSelected = selectedImageIds.contains(image.id)
        let isLocal = localImageNames.contains(image.id)

        // Visual state: On iPad (cyan) > Labeled (green) > Unlabeled (default)
        let tintColor: Color = isLocal ? .cyan : image.hasLabel ? .green : .gray
        let bgColor: Color = isSelected ? Color.accentColor.opacity(0.1)
            : isLocal ? Color.cyan.opacity(0.05)
            : image.hasLabel ? Color.green.opacity(0.05)
            : Color(white: 0.15)
        let iconName = isLocal ? "ipad.and.arrow.forward"
            : image.hasLabel ? "checkmark.circle" : "photo"
        let statusText: String? = isLocal ? "On iPad"
            : image.hasLabel ? "Submitted" : nil

        return Button {
            if isSelected {
                selectedImageIds.remove(image.id)
            } else {
                selectedImageIds.insert(image.id)
            }
        } label: {
            HStack(spacing: 12) {
                // Thumbnail placeholder
                RoundedRectangle(cornerRadius: 6)
                    .fill(tintColor.opacity(0.2))
                    .frame(width: 50, height: 50)
                    .overlay {
                        Image(systemName: iconName)
                            .foregroundColor(tintColor)
                    }

                // File info
                VStack(alignment: .leading, spacing: 4) {
                    Text(image.id)
                        .font(.subheadline)
                        .foregroundColor(.primary)
                        .lineLimit(1)

                    if let statusText {
                        Text(statusText)
                            .font(.caption2)
                            .foregroundColor(tintColor)
                    }
                }

                Spacer()

                // Label status badge
                if image.hasLabel {
                    Label("Labeled", systemImage: "checkmark.circle.fill")
                        .font(.caption2)
                        .foregroundColor(.green)
                } else {
                    Label("Unlabeled", systemImage: "circle")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }

                // Checkbox
                checkboxIcon(checked: isSelected)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(bgColor)
            .cornerRadius(10)
        }
        .buttonStyle(.plain)
    }

    // MARK: - Bottom Controls

    private var bottomControls: some View {
        Button {
            onImportSelected(selectedImageIds)
            dismiss()
        } label: {
            Label("Import (\(selectedImageIds.count) selected)", systemImage: "square.and.arrow.down")
                .font(.body.weight(.semibold))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
        }
        .buttonStyle(.borderedProminent)
        .tint(.blue)
        .disabled(selectedImageIds.isEmpty)
    }

    // MARK: - Checkbox Icon

    private func checkboxIcon(checked: Bool) -> some View {
        ZStack {
            Circle()
                .fill(Color.white)
                .frame(width: 26, height: 26)
            Image(systemName: checked ? "checkmark.circle.fill" : "circle")
                .font(.title3)
                .foregroundColor(checked ? .accentColor : Color(white: 0.75))
        }
    }
}
