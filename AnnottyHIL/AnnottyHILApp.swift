import SwiftUI

@main
struct AnnottyHILApp: App {
    @StateObject private var importCoordinator = ImportCoordinator()

    var body: some Scene {
        WindowGroup {
            MainView()
                .environmentObject(importCoordinator)
                .onOpenURL { url in
                    importCoordinator.enqueue(url)
                }
                .onAppear {
                    CacheManager.shared.clearOnLaunch()
                }
        }
    }
}
