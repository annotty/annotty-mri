import SwiftUI

/// Settings sheet for HIL server configuration
struct ServerSettingsView: View {
    @ObservedObject var settings: HILSettings
    @ObservedObject var hilViewModel: HILViewModel
    @Environment(\.dismiss) private var dismiss

    @State private var testResult: String?
    @State private var isTesting = false

    var body: some View {
        NavigationStack {
            Form {
                Section("HIL Server") {
                    Toggle("Enable HIL", isOn: $settings.isEnabled)

                    if settings.isEnabled {
                        TextField("Server URL", text: $settings.serverURL)
                            .keyboardType(.URL)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()

                        SecureField("API Key", text: $settings.apiKey)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()

                        Button(action: testConnection) {
                            HStack {
                                if isTesting {
                                    ProgressView()
                                        .scaleEffect(0.8)
                                }
                                Text(isTesting ? "Testing..." : "Test Connection")
                            }
                        }
                        .disabled(settings.serverURL.isEmpty || isTesting)

                        if let result = testResult {
                            Text(result)
                                .font(.caption)
                                .foregroundColor(result.hasPrefix("OK") ? .green : .red)
                        }
                    }
                }

                if settings.isConfigured {
                    Section("Status") {
                        HStack {
                            Circle()
                                .fill(hilViewModel.isConnected ? .green : .gray)
                                .frame(width: 10, height: 10)
                            Text(hilViewModel.isConnected ? "Connected" : "Not connected")
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("HIL Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func testConnection() {
        isTesting = true
        testResult = nil

        Task {
            do {
                let client = HILServerClient(baseURL: settings.serverURL, apiKey: settings.apiKey)
                let info = try await client.getInfo()
                testResult = "OK — \(info.totalImages) images, \(info.labeledImages) labeled"
            } catch {
                testResult = "Error: \(error.localizedDescription)"
            }
            isTesting = false
        }
    }
}
