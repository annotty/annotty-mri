import SwiftUI
import UniformTypeIdentifiers

/// Main app view with the complete UI layout
/// Layout: Left panel | Center canvas | Right panel
/// Top bar spans the full width
struct MainView: View {
    @StateObject private var viewModel = CanvasViewModel()
    @EnvironmentObject var importCoordinator: ImportCoordinator
    @StateObject private var hilSettings = HILSettings()
    @StateObject private var hilViewModel: HILViewModel

    @State private var showingExportSheet = false
    @State private var showingImagePicker = false
    @State private var showingImageSettings = false
    @State private var showingDashboard = false
    @State private var showingHILSettings = false
    @State private var isDropTargeted = false

    init() {
        let settings = HILSettings()
        _hilSettings = StateObject(wrappedValue: settings)
        _hilViewModel = StateObject(wrappedValue: HILViewModel(settings: settings))
    }

    var body: some View {
        VStack(spacing: 0) {
            // Top bar
            TopBarView(
                currentIndex: viewModel.currentImageIndex,
                totalCount: viewModel.totalImageCount,
                isLoading: viewModel.isLoading,
                isSaving: viewModel.isSaving,
                onPrevious: { viewModel.previousImage() },
                onNext: { viewModel.nextImage() },
                onGoTo: { index in viewModel.goToImage(index: index) },
                onResetView: { viewModel.resetView() },
                onClear: { viewModel.clearAllAnnotations() },
                onExport: { showingExportSheet = true },
                onLoad: { showingImagePicker = true },
                onReload: { viewModel.reloadImagesFromProject() },
                onUndo: { viewModel.undo() },
                onRedo: { viewModel.redo() },
                onDelete: { viewModel.deleteCurrentImage() },
                canUndo: viewModel.canUndo,
                canRedo: viewModel.canRedo
            )

            // Main content area
            HStack(spacing: 0) {
                // Left panel - Thickness slider
                LeftPanelView(
                    brushRadius: $viewModel.brushRadius,
                    isPainting: $viewModel.isPainting,
                    actualBrushSize: viewModel.brushPreviewSize,
                    annotationColor: viewModel.annotationColor
                )
                .frame(width: 80)

                // Center - Canvas
                CanvasContainerView(viewModel: viewModel)

                // Right panel - Color, settings button, HIL
                RightPanelView(
                    annotationColor: $viewModel.annotationColor,
                    isFillMode: $viewModel.isFillMode,
                    isSmoothMode: $viewModel.isSmoothMode,
                    isUNetLoading: viewModel.isUNetLoading,
                    isUNetProcessing: viewModel.isUNetProcessing,
                    classNames: viewModel.classNames,
                    hiddenClassIDs: $viewModel.hiddenClassIDs,
                    onSettingsTapped: {
                        withAnimation(.easeInOut(duration: 0.25)) {
                            showingImageSettings = true
                        }
                    },
                    isHILEnabled: hilSettings.isConfigured,
                    isHILSubmitting: hilViewModel.isHILSubmitting,
                    localImageCount: viewModel.totalImageCount,
                    trainingStatus: hilViewModel.trainingStatus,
                    serverInfo: hilViewModel.serverInfo,
                    onServerPredictTapped: {
                        viewModel.runUNetPrediction()
                    },
                    onSubmitTapped: {
                        Task { await hilViewModel.submitAndDelete(canvasVM: viewModel) }
                    },
                    onTrainTapped: {
                        Task { await hilViewModel.startTraining() }
                    },
                    onCancelTrainTapped: {
                        Task { await hilViewModel.cancelTraining() }
                    },
                    isSyncingModel: hilViewModel.isSyncingModel,
                    modelSource: viewModel.unetService.modelSource,
                    onSyncModelTapped: {
                        Task { await hilViewModel.syncModel(canvasVM: viewModel) }
                    },
                    hilErrorMessage: hilViewModel.errorMessage
                )
                .frame(width: 120)
            }
        }
        .background(Color(white: 0.15))
        .overlay {
            // Drop target visual feedback
            if isDropTargeted {
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.accentColor, lineWidth: 3)
                    .background(Color.accentColor.opacity(0.08))
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
            }
        }
        .onDrop(of: [UTType.image], isTargeted: $isDropTargeted) { providers in
            viewModel.handleDroppedProviders(providers)
            return true
        }
        .ignoresSafeArea(.keyboard)
        .overlay {
            // Image settings slide-in panel
            if showingImageSettings {
                ImageSettingsOverlayView(
                    isPresented: $showingImageSettings,
                    imageContrast: $viewModel.imageContrast,
                    imageBrightness: $viewModel.imageBrightness,
                    maskFillAlpha: $viewModel.maskFillAlpha,
                    maskEdgeAlpha: $viewModel.maskEdgeAlpha,
                    smoothKernelSize: $viewModel.smoothKernelSize,
                    classNames: $viewModel.classNames,
                    onClearClassNames: { viewModel.clearClassNames() },
                    onDeleteAllFiles: { viewModel.deleteAllFiles() }
                )
                .transition(.move(edge: .trailing))
            }

            // U-Net loading overlay
            if viewModel.isUNetLoading {
                UNetLoadingOverlayView()
                    .transition(.opacity)
            }
        }
        .overlay(alignment: .top) {
            // Import toast notification
            if importCoordinator.showToast {
                ImportToastView(count: importCoordinator.importedCount)
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .padding(.top, 60)
            }
        }
        .animation(.easeInOut(duration: 0.3), value: importCoordinator.showToast)
        .sheet(isPresented: $showingImagePicker) {
            ImagePickerView(
                onImageSelected: { url in
                    viewModel.importImage(from: url)
                },
                onImagesSelected: { urls in
                    viewModel.importImages(from: urls)
                },
                onFolderSelected: { url in
                    viewModel.importImagesFromFolder(url)
                },
                onProjectSelected: { url in
                    viewModel.openProject(at: url)
                },
                onCloudflareSettingsTapped: {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                        showingHILSettings = true
                    }
                },
                onHILDashboardTapped: {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                        showingDashboard = true
                    }
                },
                isHILConfigured: hilSettings.isConfigured
            )
        }
        .sheet(isPresented: $showingExportSheet) {
            ExportSheetView(viewModel: viewModel)
        }
        .sheet(isPresented: $showingHILSettings) {
            ServerSettingsView(settings: hilSettings, hilViewModel: hilViewModel)
        }
        .sheet(isPresented: $showingDashboard) {
            HILDashboardView(
                hilViewModel: hilViewModel,
                settings: hilSettings,
                localImageNames: viewModel.localImageFileNames,
                onImportSelected: { imageIds in
                    Task {
                        await hilViewModel.importImages(imageIds: imageIds, canvasVM: viewModel)
                    }
                }
            )
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)) { _ in
            // Save when app goes to background
            viewModel.saveBeforeBackground()
        }
    }
}

// MARK: - Left Panel

struct LeftPanelView: View {
    @Binding var brushRadius: Float
    @Binding var isPainting: Bool
    var actualBrushSize: CGFloat  // Actual size considering zoom level
    var annotationColor: Color    // Current annotation color

    /// Maximum display size for brush preview (in points)
    private let maxPreviewSize: CGFloat = 60

    /// Preview size (shows actual brush size, clamped to fit)
    private var previewSize: CGFloat {
        min(actualBrushSize, maxPreviewSize)
    }

    /// Scale factor shown when brush is larger than preview area
    private var displayScale: String? {
        if actualBrushSize > maxPreviewSize {
            let ratio = actualBrushSize / maxPreviewSize
            return String(format: "x%.1f", ratio)
        }
        return nil
    }

    var body: some View {
        VStack(spacing: 16) {
            // Paint/Erase toggle
            VStack(spacing: 8) {
                Button(action: { isPainting = true }) {
                    Image(systemName: "pencil.tip")
                        .font(.title2)
                        .foregroundColor(isPainting ? .green : .gray)
                }
                .buttonStyle(.plain)

                Button(action: { isPainting = false }) {
                    Image(systemName: "eraser.fill")
                        .font(.title2)
                        .foregroundColor(!isPainting ? .red : .gray)
                }
                .buttonStyle(.plain)
            }
            .padding(.top, 20)

            Spacer()

            // Brush size preview circle (reflects actual drawn size)
            ZStack {
                // Background circle (shows max preview area)
                Circle()
                    .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                    .frame(width: maxPreviewSize, height: maxPreviewSize)

                // Actual brush size circle (uses annotation color when painting, red when erasing)
                Circle()
                    .fill(isPainting ? annotationColor.opacity(0.5) : Color.red.opacity(0.5))
                    .frame(width: previewSize, height: previewSize)

                Circle()
                    .stroke(isPainting ? annotationColor : Color.red, lineWidth: 2)
                    .frame(width: previewSize, height: previewSize)

                // Show scale indicator if brush is larger than preview
                if let scale = displayScale {
                    Text(scale)
                        .font(.caption2)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                }
            }
            .frame(width: maxPreviewSize, height: maxPreviewSize)
            .animation(.easeOut(duration: 0.1), value: previewSize)

            // Brush size number
            Text("\(Int(brushRadius))")
                .font(.caption)
                .foregroundColor(.white)

            // Thickness slider (vertical)
            ThicknessSliderView(radius: $brushRadius)
                .frame(height: 250)

            Spacer()
        }
        .frame(maxHeight: .infinity)
        .background(Color(white: 0.1))
    }
}

// MARK: - U-Net Loading Overlay

struct UNetLoadingOverlayView: View {
    var body: some View {
        ZStack {
            // Semi-transparent background
            Color.black.opacity(0.6)
                .ignoresSafeArea()

            // Loading card
            VStack(spacing: 20) {
                // Animated icon
                Image(systemName: "wand.and.stars")
                    .font(.system(size: 48))
                    .foregroundColor(.cyan)
                    .symbolEffect(.pulse)

                // Title
                Text("Loading U-Net")
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)

                // Progress indicator
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle(tint: .cyan))
                    .scaleEffect(1.2)

                // Description
                Text("Preparing segmentation models...")
                    .font(.subheadline)
                    .foregroundColor(.gray)
                    .multilineTextAlignment(.center)
            }
            .padding(40)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(Color(white: 0.15))
                    .shadow(color: .black.opacity(0.3), radius: 20)
            )
        }
    }
}

// MARK: - Import Toast

struct ImportToastView: View {
    let count: Int

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.body)
            Text("\(count) image\(count == 1 ? "" : "s") imported")
                .font(.subheadline)
                .fontWeight(.medium)
        }
        .foregroundColor(.white)
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(
            Capsule()
                .fill(Color.accentColor.opacity(0.9))
                .shadow(color: .black.opacity(0.3), radius: 8, y: 4)
        )
    }
}

// MARK: - Preview

#Preview {
    MainView()
        .environmentObject(ImportCoordinator())
}
