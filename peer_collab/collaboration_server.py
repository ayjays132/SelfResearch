
from flask import Flask, request, jsonify
import torch
from collections import defaultdict
import time

class CollaborationServer:
    """
    A Flask-based REST API for multi-user collaboration on research projects.
    Includes functionality for managing shared notes, feedback, and simulated real-time updates.
    """
    def __init__(self, host='0.0.0.0', port=5000, device='cpu'):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.device = torch.device(device)
        self.shared_notes = defaultdict(str) # project_id -> notes content
        self.feedback_data = defaultdict(list) # project_id -> list of feedback items
        self.project_members = defaultdict(set) # project_id -> set of user_ids
        self.project_roles = defaultdict(dict) # project_id -> {user_id -> role}
        self._setup_routes()
        print(f"CollaborationServer initialized on device: {self.device}")

    def _setup_routes(self):
        @self.app.route('/notes/<project_id>', methods=['GET', 'POST'])
        def handle_notes(project_id):
            # In a real app, you'd verify user authentication and permissions here
            if request.method == 'GET':
                # Simulate some PyTorch computation
                dummy_tensor = torch.randn(3, 3, device=self.device)
                _ = torch.inverse(dummy_tensor) if dummy_tensor.det() != 0 else dummy_tensor # Dummy computation
                return jsonify({'notes': self.shared_notes[project_id]})
            elif request.method == 'POST':
                data = request.json
                notes = data.get('notes', '')
                user_id = data.get('user_id', 'anonymous') # Simulate user
                self.shared_notes[project_id] = notes
                # Simulate some PyTorch computation
                dummy_tensor = torch.tensor([len(notes)], dtype=torch.float32, device=self.device)
                _ = torch.log(dummy_tensor + 1) # Dummy computation
                print(f"Project {project_id} notes updated by {user_id}")
                return jsonify({'message': 'Notes updated successfully', 'notes': notes})

        @self.app.route('/feedback/<project_id>', methods=['GET', 'POST'])
        def handle_feedback(project_id):
            if request.method == 'GET':
                # Simulate some PyTorch computation
                dummy_tensor = torch.randn(2, 2, device=self.device)
                _ = torch.exp(dummy_tensor) # Dummy computation
                return jsonify({'feedback': self.feedback_data[project_id]})
            elif request.method == 'POST':
                data = request.json
                feedback_item = {
                    "user_id": data.get('user_id', 'anonymous'),
                    "comment": data.get('comment'),
                    "timestamp": time.time()
                }
                self.feedback_data[project_id].append(feedback_item)
                # Simulate some PyTorch computation
                dummy_tensor = torch.tensor([len(self.feedback_data[project_id])], dtype=torch.float32, device=self.device)
                _ = torch.sin(dummy_tensor) # Dummy computation
                print(f"Feedback added to project {project_id} by {feedback_item['user_id']}")
                return jsonify({'message': 'Feedback added successfully', 'feedback': feedback_item})

        @self.app.route('/projects/<project_id>/members', methods=['GET', 'POST', 'DELETE'])
        def manage_members(project_id):
            if request.method == 'GET':
                return jsonify({'members': list(self.project_members[project_id])})
            elif request.method == 'POST':
                data = request.json
                user_id = data.get('user_id')
                role = data.get('role', 'member')
                if user_id:
                    self.project_members[project_id].add(user_id)
                    self.project_roles[project_id][user_id] = role
                    print(f"User {user_id} added to project {project_id} with role {role}")
                    return jsonify({'message': 'Member added', 'user_id': user_id, 'role': role})
                return jsonify({'message': 'User ID required'}), 400
            elif request.method == 'DELETE':
                data = request.json
                user_id = data.get('user_id')
                if user_id and user_id in self.project_members[project_id]:
                    self.project_members[project_id].remove(user_id)
                    if user_id in self.project_roles[project_id]:
                        del self.project_roles[project_id][user_id]
                    print(f"User {user_id} removed from project {project_id}")
                    return jsonify({'message': 'Member removed', 'user_id': user_id})
                return jsonify({'message': 'User ID not found or required'}), 400

    def run(self):
        print(f"Starting Flask server on {self.host}:{self.port}")
        # In a production environment, use a WSGI server like Gunicorn
        self.app.run(host=self.host, port=self.port)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available! Using GPU.")
    else:
        device = 'cpu'
        print("CUDA not available. Using CPU.")

    server = CollaborationServer(device=device)
    print("To run the server, uncomment `server.run()` and execute this file.")
    print("Example usage with curl:")
    print("\n--- Notes API ---")
    print("GET notes: curl http://127.0.0.1:5000/notes/project_alpha")
    print("POST notes: curl -X POST -H \"Content-Type: application/json\" -d '{\"user_id\": \"user1\", \"notes\": \"Initial research notes for project Alpha.\"}' http://127.0.0.1:5000/notes/project_alpha")
    print("\n--- Feedback API ---")
    print("GET feedback: curl http://127.0.0.1:5000/feedback/project_alpha")
    print("POST feedback: curl -X POST -H \"Content-Type: application/json\" -d '{\"user_id\": \"user2\", \"comment\": \"Great progress on the literature review!\"}' http://127.0.0.1:5000/feedback/project_alpha")
    print("\n--- Members API ---")
    print("GET members: curl http://127.0.0.1:5000/projects/project_alpha/members")
    print("POST member: curl -X POST -H \"Content-Type: application/json\" -d '{\"user_id\": \"user3\", \"role\": \"contributor\"}' http://127.0.0.1:5000/projects/project_alpha/members")
    print("DELETE member: curl -X DELETE -H \"Content-Type: application/json\" -d '{\"user_id\": \"user3\"}' http://127.0.0.1:5000/projects/project_alpha/members")

    # server.run() # Uncomment to run the server directly for testing


