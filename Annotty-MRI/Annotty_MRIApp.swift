import SwiftUI

@main
struct Annotty_MRIApp: App {
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
