//import Foundation
//import Network
//
//final class SocketClient {
//
//    private var connection: NWConnection?
//    private let queue = DispatchQueue(label: "socket.queue")
//    private var isWaiting = false
//
//    private let host: NWEndpoint.Host
//    private let port: NWEndpoint.Port
//
//    init(host: String, port: UInt16) {
//        self.host = NWEndpoint.Host(host)
//        self.port = NWEndpoint.Port(rawValue: port)!
//    }
//
//    func connect() {
//        guard connection == nil else { return }
//
//        let conn = NWConnection(host: host, port: port, using: .tcp)
//        connection = conn
//
//        conn.stateUpdateHandler = { state in
//            if case .ready = state {
//                print("✅ Socket connected")
//            }
//        }
//        conn.start(queue: queue)
//    }
//
//    func disconnect() {
//        connection?.cancel()
//        connection = nil
//    }
//
//    func sendRequest(
//        _ json: [String: Any],
//        completion: @escaping (Result<[String: Any], Error>) -> Void
//    ) {
//        guard let conn = connection, !isWaiting else { return }
//        isWaiting = true
//
//        let payload = try! JSONSerialization.data(withJSONObject: json)
//        var frame = Data()
//        var len = UInt32(payload.count).bigEndian
//        frame.append(Data(bytes: &len, count: 4))
//        frame.append(payload)
//
//        conn.send(content: frame, completion: .contentProcessed { _ in
//            self.receiveResponse(conn, completion: completion)
//        })
//    }
//
//    private func receiveResponse(
//        _ conn: NWConnection,
//        completion: @escaping (Result<[String: Any], Error>) -> Void
//    ) {
//        conn.receive(minimumIncompleteLength: 4, maximumLength: 4) { data, _, _, _ in
//            let len = data!.withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
//
//            conn.receive(minimumIncompleteLength: Int(len), maximumLength: Int(len)) { payload, _, _, _ in
//                self.isWaiting = false
//                let obj = try! JSONSerialization.jsonObject(with: payload!)
//                completion(.success(obj as! [String: Any]))
//            }
//        }
//    }
//}
