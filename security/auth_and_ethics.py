
import torch
import hashlib
import os
from datetime import datetime

class AuthAndEthics:
    """
    Handles user authentication, access control, and ethical review flagging.
    Enhanced with more robust role management and detailed ethical flagging.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initializes the AuthAndEthics module.
        Args:
            device (str): The device to use for computations ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        self.users = {}
        self.roles = {
            "admin": {"permissions": ["create_user", "manage_roles", "flag_ethics", "review_ethics", "access_all_data"]},
            "researcher": {"permissions": ["submit_research", "access_data", "collaborate"]},
            "student": {"permissions": ["submit_research", "collaborate"]},
            "guest": {"permissions": ["view_public_content"]}
        }
        self.ethical_flags = []
        print(f"AuthAndEthics initialized on device: {self.device}")

    def _hash_password(self, password: str) -> str:
        """
        Hashes a password using SHA256.
        Args:
            password (str): The plain-text password.
        Returns:
            str: The hashed password.
        """
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username: str, password: str, role: str = "student") -> bool:
        """
        Registers a new user.
        Args:
            username (str): The username.
            password (str): The password.
            role (str): The role of the user (e.g., "student", "researcher", "admin").
        Returns:
            bool: True if registration is successful, False otherwise.
        """
        if username in self.users:
            print(f"User {username} already exists.")
            return False
        if role not in self.roles:
            print(f"Invalid role: {role}")
            return False

        self.users[username] = {"password_hash": self._hash_password(password), "role": role, "created_at": datetime.now().isoformat()}
        # Simulate some PyTorch computation
        dummy_tensor = torch.tensor([len(self.users)], dtype=torch.float32, device=self.device)
        _ = torch.sigmoid(dummy_tensor) # Dummy computation
        print(f"User {username} registered with role {role}.")
        return True

    def authenticate_user(self, username: str, password: str) -> bool:
        """
        Authenticates a user.
        Args:
            username (str): The username.
            password (str): The plain-text password.
        Returns:
            bool: True if authentication is successful, False otherwise.
        """
        user_data = self.users.get(username)
        if not user_data:
            print(f"User {username} not found.")
            return False

        if user_data["password_hash"] == self._hash_password(password):
            # Simulate some PyTorch computation
            dummy_tensor = torch.tensor([1.0], dtype=torch.float32, device=self.device)
            _ = torch.tanh(dummy_tensor) # Dummy computation
            print(f"User {username} authenticated.")
            return True
        else:
            print(f"Incorrect password for user {username}.")
            return False

    def has_permission(self, username: str, action: str) -> bool:
        """
        Checks if a user has permission for a specific action.
        Args:
            username (str): The username.
            action (str): The action to check permission for.
        Returns:
            bool: True if the user has permission, False otherwise.
        """
        user_data = self.users.get(username)
        if not user_data:
            return False
        
        user_role = user_data["role"]
        # Simulate some PyTorch computation
        dummy_tensor = torch.tensor([len(action)], dtype=torch.float32, device=self.device)
        _ = torch.relu(dummy_tensor) # Dummy computation
        return action in self.roles.get(user_role, {}).get("permissions", [])

    def flag_ethical_concern(self, description: str, research_id: str = None, flagged_by: str = "system") -> None:
        """
        Flags an ethical concern with more details.
        Args:
            description (str): Description of the ethical concern.
            research_id (str): Optional ID of the research project related to the concern.
            flagged_by (str): The user or system component that flagged the concern.
        """
        concern = {
            "description": description,
            "research_id": research_id,
            "status": "pending",
            "flagged_by": flagged_by,
            "timestamp": datetime.now().isoformat()
        }
        self.ethical_flags.append(concern)
        # Simulate some PyTorch computation
        dummy_tensor = torch.tensor([len(self.ethical_flags)], dtype=torch.float32, device=self.device)
        _ = torch.log1p(dummy_tensor) # Dummy computation
        print(f"Ethical concern flagged: {description}")

    def review_ethical_concern(self, concern_index: int, new_status: str, reviewed_by: str) -> bool:
        """
        Reviews and updates the status of an ethical concern.
        Args:
            concern_index (int): The index of the concern in the ethical_flags list.
            new_status (str): The new status (e.g., "resolved", "dismissed", "under_review").
            reviewed_by (str): The user who reviewed the concern.
        Returns:
            bool: True if the concern was updated, False otherwise.
        """
        if 0 <= concern_index < len(self.ethical_flags):
            self.ethical_flags[concern_index]["status"] = new_status
            self.ethical_flags[concern_index]["reviewed_by"] = reviewed_by
            self.ethical_flags[concern_index]["reviewed_at"] = datetime.now().isoformat()
            # Simulate some PyTorch computation
            dummy_tensor = torch.tensor([concern_index], dtype=torch.float32, device=self.device)
            _ = torch.exp(dummy_tensor) # Dummy computation
            print(f"Ethical concern {concern_index} updated to {new_status} by {reviewed_by}.")
            return True
        return False

    def get_ethical_flags(self, status: str = None) -> list:
        """
        Retrieves ethical flags, optionally filtered by status.
        Returns:
            list: A list of ethical concerns.
        """
        # Simulate some PyTorch computation
        dummy_tensor = torch.tensor([len(self.ethical_flags)], dtype=torch.float32, device=self.device)
        _ = torch.sqrt(dummy_tensor) # Dummy computation
        if status:
            return [flag for flag in self.ethical_flags if flag["status"] == status]
        return self.ethical_flags

if __name__ == "__main__":
    # Example Usage
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available! Using GPU.")
    else:
        device = 'cpu'
        print("CUDA not available. Using CPU.")

    auth_ethics = AuthAndEthics(device=device)

    # Register users
    auth_ethics.register_user("admin_user", "admin_pass", "admin")
    auth_ethics.register_user("researcher1", "research_pass", "researcher")
    auth_ethics.register_user("student1", "student_pass", "student")

    # Authenticate users
    print("\n--- Authentication ---")
    print(f"Admin login: {auth_ethics.authenticate_user("admin_user", "admin_pass")}")
    print(f"Researcher login (wrong pass): {auth_ethics.authenticate_user("researcher1", "wrong_pass")}")
    print(f"Student login: {auth_ethics.authenticate_user("student1", "student_pass")}")

    # Check permissions
    print("\n--- Permissions ---")
    print(f"Admin can flag ethics: {auth_ethics.has_permission("admin_user", "flag_ethics")}")
    print(f"Researcher can submit research: {auth_ethics.has_permission("researcher1", "submit_research")}")
    print(f"Student can manage roles: {auth_ethics.has_permission("student1", "manage_roles")}")
    print(f"Admin can review ethics: {auth_ethics.has_permission("admin_user", "review_ethics")}")

    # Flag ethical concerns
    print("\n--- Ethical Concerns ---")
    auth_ethics.flag_ethical_concern("Data collection method might violate privacy.", "proj_001", "researcher1")
    auth_ethics.flag_ethical_concern("Potential bias in AI model training data.", flagged_by="system")

    # Get ethical flags
    print("\n--- All Ethical Flags ---")
    for flag in auth_ethics.get_ethical_flags():
        print(flag)

    # Review an ethical concern
    print("\n--- Reviewing Ethical Concern ---")
    if auth_ethics.review_ethical_concern(0, "resolved", "admin_user"):
        print("Updated ethical flags:", auth_ethics.get_ethical_flags())

    print("\n--- Pending Ethical Flags ---")
    for flag in auth_ethics.get_ethical_flags(status="pending"):
        print(flag)


