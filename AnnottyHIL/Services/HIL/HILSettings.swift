import Foundation
import Combine

/// Persistent settings for HIL server connection
class HILSettings: ObservableObject {
    @Published var serverURL: String
    @Published var apiKey: String
    @Published var isEnabled: Bool

    /// Whether the server URL has been configured
    var isConfigured: Bool {
        !serverURL.isEmpty && isEnabled
    }

    private var cancellables = Set<AnyCancellable>()

    init() {
        self.serverURL = UserDefaults.standard.string(forKey: "hil_server_url") ?? ""
        self.apiKey = UserDefaults.standard.string(forKey: "hil_api_key") ?? ""
        self.isEnabled = UserDefaults.standard.bool(forKey: "hil_enabled")

        $serverURL
            .dropFirst()
            .sink { UserDefaults.standard.set($0, forKey: "hil_server_url") }
            .store(in: &cancellables)

        $apiKey
            .dropFirst()
            .sink { UserDefaults.standard.set($0, forKey: "hil_api_key") }
            .store(in: &cancellables)

        $isEnabled
            .dropFirst()
            .sink { UserDefaults.standard.set($0, forKey: "hil_enabled") }
            .store(in: &cancellables)
    }
}
