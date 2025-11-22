"""
End-to-End Testing Suite for PP Frontend Interface
Tests complete user workflows from submission to results viewing
"""

import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class E2ETestSuite:
    """End-to-end test suite for the PP application"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.results: List[Dict[str, Any]] = []
        self.test_data = {
            "prediction_id": None,
            "campaign_id": None,
        }
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with color coding"""
        color = Colors.BLUE if level == "INFO" else Colors.YELLOW if level == "WARN" else Colors.RED
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{color}[{timestamp}] {level}: {message}{Colors.RESET}")
        
    def success(self, message: str):
        """Log a success message"""
        print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")
        
    def failure(self, message: str):
        """Log a failure message"""
        print(f"{Colors.RED}✗ {message}{Colors.RESET}")
        
    def section(self, message: str):
        """Log a section header"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
        print(f"{message}")
        print(f"{'='*80}{Colors.RESET}\n")
        
    def record_result(self, test_name: str, passed: bool, message: str = "", duration: float = 0):
        """Record test result"""
        self.results.append({
            "test": test_name,
            "passed": passed,
            "message": message,
            "duration": duration
        })
        
    def test_health_check(self) -> bool:
        """Test 1: Health check endpoint"""
        self.section("Test 1: Health Check")
        start = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                self.success(f"Health check passed: {data}")
                self.record_result("health_check", True, "", duration)
                return True
            else:
                self.failure(f"Health check failed: {response.status_code}")
                self.record_result("health_check", False, f"Status {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.failure(f"Health check error: {e}")
            self.record_result("health_check", False, str(e), duration)
            return False
            
    def test_prediction_submission(self) -> bool:
        """Test 2: Submit a new prediction"""
        self.section("Test 2: Prediction Submission")
        start = time.time()
        
        payload = {
            "protein_sequence": "ACDEFGHIKLMNPQRSTVWY",
            "config": {
                "iterations": 100,
                "agents": 3,
                "preset": "balanced"
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/predictions",
                json=payload,
                timeout=10
            )
            duration = time.time() - start
            
            if response.status_code in [200, 201]:
                data = response.json()
                self.test_data["prediction_id"] = data.get("id")
                self.success(f"Prediction submitted: ID={self.test_data['prediction_id']}")
                self.log(f"Response: {json.dumps(data, indent=2)}")
                self.record_result("prediction_submission", True, "", duration)
                return True
            else:
                self.failure(f"Prediction submission failed: {response.status_code}")
                self.log(f"Response: {response.text}")
                self.record_result("prediction_submission", False, f"Status {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.failure(f"Prediction submission error: {e}")
            self.record_result("prediction_submission", False, str(e), duration)
            return False
            
    def test_prediction_status(self) -> bool:
        """Test 3: Check prediction status"""
        self.section("Test 3: Prediction Status")
        
        if not self.test_data["prediction_id"]:
            self.failure("No prediction ID available")
            self.record_result("prediction_status", False, "No prediction ID")
            return False
            
        start = time.time()
        
        try:
            pred_id = self.test_data["prediction_id"]
            response = requests.get(
                f"{self.api_url}/predictions/{pred_id}",
                timeout=5
            )
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                self.success(f"Prediction status: {status}")
                self.log(f"Progress: {data.get('progress', 0)}%")
                self.record_result("prediction_status", True, "", duration)
                return True
            else:
                self.failure(f"Status check failed: {response.status_code}")
                self.record_result("prediction_status", False, f"Status {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.failure(f"Status check error: {e}")
            self.record_result("prediction_status", False, str(e), duration)
            return False
            
    def test_prediction_list(self) -> bool:
        """Test 4: List all predictions"""
        self.section("Test 4: Prediction List")
        start = time.time()
        
        try:
            response = requests.get(
                f"{self.api_url}/predictions",
                params={"limit": 10},
                timeout=5
            )
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                count = len(data.get("items", []))
                self.success(f"Retrieved {count} predictions")
                self.record_result("prediction_list", True, "", duration)
                return True
            else:
                self.failure(f"List retrieval failed: {response.status_code}")
                self.record_result("prediction_list", False, f"Status {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.failure(f"List retrieval error: {e}")
            self.record_result("prediction_list", False, str(e), duration)
            return False
            
    def test_campaign_creation(self) -> bool:
        """Test 5: Create a campaign"""
        self.section("Test 5: Campaign Creation")
        start = time.time()
        
        payload = {
            "name": "E2E Test Campaign",
            "protein_ids": ["1UBQ", "1CRN"],
            "phases": 2,
            "iterations_per_phase": 50
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/campaigns",
                json=payload,
                timeout=10
            )
            duration = time.time() - start
            
            if response.status_code in [200, 201]:
                data = response.json()
                self.test_data["campaign_id"] = data.get("id")
                self.success(f"Campaign created: ID={self.test_data['campaign_id']}")
                self.record_result("campaign_creation", True, "", duration)
                return True
            else:
                self.failure(f"Campaign creation failed: {response.status_code}")
                self.log(f"Response: {response.text}")
                self.record_result("campaign_creation", False, f"Status {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.failure(f"Campaign creation error: {e}")
            self.record_result("campaign_creation", False, str(e), duration)
            return False
            
    def test_results_retrieval(self) -> bool:
        """Test 6: Retrieve prediction results"""
        self.section("Test 6: Results Retrieval")
        
        if not self.test_data["prediction_id"]:
            self.failure("No prediction ID available")
            self.record_result("results_retrieval", False, "No prediction ID")
            return False
            
        start = time.time()
        
        try:
            pred_id = self.test_data["prediction_id"]
            response = requests.get(
                f"{self.api_url}/results/{pred_id}",
                timeout=5
            )
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                self.success("Results retrieved successfully")
                self.log(f"RMSD: {data.get('final_rmsd', 'N/A')}")
                self.log(f"Energy: {data.get('final_energy', 'N/A')}")
                self.record_result("results_retrieval", True, "", duration)
                return True
            elif response.status_code == 404:
                self.log("Results not yet available (prediction may still be running)", "WARN")
                self.record_result("results_retrieval", True, "Not yet available", duration)
                return True
            else:
                self.failure(f"Results retrieval failed: {response.status_code}")
                self.record_result("results_retrieval", False, f"Status {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.failure(f"Results retrieval error: {e}")
            self.record_result("results_retrieval", False, str(e), duration)
            return False
            
    def test_websocket_connection(self) -> bool:
        """Test 7: WebSocket connectivity (basic check)"""
        self.section("Test 7: WebSocket Connection")
        start = time.time()
        
        # For now, just check if Socket.IO endpoint responds
        try:
            response = requests.get(
                f"{self.base_url}/socket.io/",
                timeout=5,
                allow_redirects=False
            )
            duration = time.time() - start
            
            # Socket.IO should respond with upgrade required or similar
            if response.status_code in [200, 400, 426]:
                self.success("WebSocket endpoint is accessible")
                self.record_result("websocket_connection", True, "", duration)
                return True
            else:
                self.failure(f"WebSocket endpoint returned: {response.status_code}")
                self.record_result("websocket_connection", False, f"Status {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.failure(f"WebSocket check error: {e}")
            self.record_result("websocket_connection", False, str(e), duration)
            return False
            
    def test_api_documentation(self) -> bool:
        """Test 8: API documentation availability"""
        self.section("Test 8: API Documentation")
        start = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/docs", timeout=5)
            duration = time.time() - start
            
            if response.status_code == 200:
                self.success("API documentation is accessible")
                self.record_result("api_documentation", True, "", duration)
                return True
            else:
                self.failure(f"Documentation failed: {response.status_code}")
                self.record_result("api_documentation", False, f"Status {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.failure(f"Documentation check error: {e}")
            self.record_result("api_documentation", False, str(e), duration)
            return False
            
    def generate_report(self):
        """Generate test report"""
        self.section("Test Report")
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        total_duration = sum(r["duration"] for r in self.results)
        
        print(f"{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"  Total tests: {total}")
        print(f"  {Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"  {Colors.RED}Failed: {failed}{Colors.RESET}")
        print(f"  Total duration: {total_duration:.2f}s")
        print()
        
        if failed > 0:
            print(f"{Colors.BOLD}Failed Tests:{Colors.RESET}")
            for result in self.results:
                if not result["passed"]:
                    print(f"  {Colors.RED}✗ {result['test']}{Colors.RESET}")
                    if result["message"]:
                        print(f"    {result['message']}")
            print()
            
        # Save report to file
        report_path = Path(__file__).parent / "e2e_test_report.json"
        with open(report_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "duration": total_duration
                },
                "results": self.results
            }, f, indent=2)
            
        self.log(f"Report saved to: {report_path}")
        
        return failed == 0
        
    def run_all_tests(self) -> bool:
        """Run all E2E tests"""
        print(f"{Colors.BOLD}{Colors.BLUE}")
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                    PP Frontend Interface E2E Test Suite                       ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.RESET}")
        
        self.log(f"Testing against: {self.base_url}")
        
        # Run tests in sequence
        tests = [
            self.test_health_check,
            self.test_prediction_submission,
            self.test_prediction_status,
            self.test_prediction_list,
            self.test_campaign_creation,
            self.test_results_retrieval,
            self.test_websocket_connection,
            self.test_api_documentation,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                self.failure(f"Unexpected error in {test.__name__}: {e}")
                self.record_result(test.__name__, False, str(e), 0)
            time.sleep(0.5)  # Brief pause between tests
            
        # Generate report
        return self.generate_report()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run E2E tests for PP Frontend Interface")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL for the backend API (default: http://localhost:8000)"
    )
    args = parser.parse_args()
    
    suite = E2ETestSuite(base_url=args.url)
    success = suite.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
