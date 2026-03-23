import SwiftUI

/// Top bar with image navigation and export button
struct TopBarView: View {
    let currentIndex: Int
    let totalCount: Int
    let isLoading: Bool
    let isSaving: Bool
    let onPrevious: () -> Void
    let onNext: () -> Void
    let onGoTo: (Int) -> Void
    let onResetView: () -> Void
    let onClear: () -> Void
    let onExport: () -> Void
    let onLoad: () -> Void
    let onReload: () -> Void
    let onUndo: () -> Void
    let onRedo: () -> Void
    let onDelete: () -> Void
    let canUndo: Bool
    let canRedo: Bool

    @State private var showingDeleteAlert = false

    var body: some View {
        HStack {
            // Load button
            Button(action: onLoad) {
                HStack(spacing: 4) {
                    Image(systemName: "folder")
                    Text("Load")
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.blue.opacity(0.8))
                .foregroundColor(.white)
                .cornerRadius(8)
            }
            .buttonStyle(.plain)

            // Reload button
            Button(action: onReload) {
                Image(systemName: "arrow.clockwise")
                    .padding(8)
                    .background(Color.gray.opacity(0.5))
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }
            .buttonStyle(.plain)

            // Undo button
            Button(action: onUndo) {
                Image(systemName: "arrow.uturn.backward")
                    .padding(8)
                    .background(canUndo ? Color.gray.opacity(0.5) : Color.gray.opacity(0.2))
                    .foregroundColor(canUndo ? .white : .gray)
                    .cornerRadius(8)
            }
            .buttonStyle(.plain)
            .disabled(!canUndo)

            // Redo button
            Button(action: onRedo) {
                Image(systemName: "arrow.uturn.forward")
                    .padding(8)
                    .background(canRedo ? Color.gray.opacity(0.5) : Color.gray.opacity(0.2))
                    .foregroundColor(canRedo ? .white : .gray)
                    .cornerRadius(8)
            }
            .buttonStyle(.plain)
            .disabled(!canRedo)

            Spacer()

            // Image navigation
            ImageNavigatorView(
                currentIndex: currentIndex,
                totalCount: totalCount,
                onPrevious: onPrevious,
                onNext: onNext,
                onGoTo: onGoTo
            )

            // Loading/Saving indicator (fixed width to prevent layout shift)
            HStack(spacing: 6) {
                if isLoading || isSaving {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .scaleEffect(0.8)
                    Text(isLoading ? "Loading..." : "Saving...")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
            }
            .frame(width: 100, alignment: .leading)

            Spacer()

            // Fit view button (reset pan/zoom/rotation)
            Button(action: onResetView) {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                    Text("Fit")
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.gray.opacity(0.6))
                .foregroundColor(.white)
                .cornerRadius(8)
            }
            .buttonStyle(.plain)
            .disabled(totalCount == 0)

            // Delete image button
            Button(action: { showingDeleteAlert = true }) {
                HStack(spacing: 4) {
                    Image(systemName: "xmark.bin")
                    Text("Delete")
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.red.opacity(0.9))
                .foregroundColor(.white)
                .cornerRadius(8)
            }
            .buttonStyle(.plain)
            .disabled(totalCount == 0)

            // Clear annotation button
            Button(action: onClear) {
                HStack(spacing: 4) {
                    Image(systemName: "trash")
                    Text("Clear")
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.red.opacity(0.7))
                .foregroundColor(.white)
                .cornerRadius(8)
            }
            .buttonStyle(.plain)
            .disabled(totalCount == 0)

            // Export button
            Button(action: onExport) {
                HStack(spacing: 4) {
                    Image(systemName: "square.and.arrow.up")
                    Text("Export")
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.green.opacity(0.8))
                .foregroundColor(.white)
                .cornerRadius(8)
            }
            .buttonStyle(.plain)
            .disabled(totalCount == 0)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color(white: 0.1))
        .alert("Delete Image", isPresented: $showingDeleteAlert) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                onDelete()
            }
        } message: {
            Text("This image and its annotation will be permanently deleted.")
        }
    }
}

// MARK: - Preview

#Preview {
    TopBarView(
        currentIndex: 12,
        totalCount: 128,
        isLoading: false,
        isSaving: false,
        onPrevious: {},
        onNext: {},
        onGoTo: { _ in },
        onResetView: {},
        onClear: {},
        onExport: {},
        onLoad: {},
        onReload: {},
        onUndo: {},
        onRedo: {},
        onDelete: {},
        canUndo: true,
        canRedo: false
    )
}
