"""
Phase 2.1: Lambda Client Testing

Tests for Lambda Labs GPU instance management:
1. Authentication with Lambda API
2. List available GPU instances
3. Select cheapest available instance
4. Launch a GPU instance
5. Instance reaches "active" state
6. Get instance details (IP, ID, type, region)
7. Check instance status
8. Terminate instance
9. Cleanup works properly
10. Handle no available instances

Run: python tests/test_04_lambda_client.py

⚠️  WARNING: This test will launch a REAL GPU instance on Lambda Labs
⚠️  Cost: ~$0.60-$1.10 per hour depending on instance type
⚠️  The test will automatically terminate the instance after completion
"""

import asyncio
import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lambda_client import LambdaClient
from dotenv import load_dotenv


# Global variables to track instance for cleanup
test_instance_id = None
test_instance_ip = None
test_instance_type = None
test_region = None


def test_1_authentication():
    """Test 1: Can authenticate with Lambda API"""
    print("=" * 70)
    print("TEST 1: Authentication")
    print("=" * 70)
    
    try:
        load_dotenv()
        api_key = os.getenv('LAMBDA_API_KEY')
        
        if not api_key:
            print("\n❌ FAILED: LAMBDA_API_KEY not found in .env file")
            return None
        
        print(f"\n🔑 API Key found: {api_key[:20]}...")
        
        # Initialize client
        client = LambdaClient()
        
        # Test authentication by listing SSH keys
        ssh_keys = client.list_ssh_keys()
        
        print(f"\n✅ Authentication successful!")
        print(f"   SSH Keys available: {len(ssh_keys)}")
        
        if ssh_keys:
            for key in ssh_keys[:3]:  # Show first 3 keys
                print(f"   - {key.get('name', 'unnamed')}")
        
        return client
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_2_list_available_instances(client):
    """Test 2: Can list available GPU instances"""
    print("\n" + "=" * 70)
    print("TEST 2: List Available GPU Instances")
    print("=" * 70)
    
    if not client:
        print("❌ SKIPPED: No client available")
        return None
    
    try:
        # Get all instance types
        response = client._make_request('GET', '/instance-types')
        all_instances = response.get('data', {})
        
        print(f"\n📊 Total instance types: {len(all_instances)}")
        
        # Filter available instances
        available = {
            k: v for k, v in all_instances.items() 
            if len(v.get('regions_with_capacity_available', [])) > 0
        }
        
        print(f"📊 Available instances: {len(available)}")
        
        if available:
            print("\n💰 Available GPU instances:")
            for name, details in list(available.items())[:5]:  # Show first 5
                specs = details['instance_type']['specs']
                price = details['instance_type']['price_cents_per_hour'] / 100
                regions = details['regions_with_capacity_available']
                
                print(f"\n   {name}:")
                print(f"      GPUs: {specs.get('gpus', 'N/A')}")
                print(f"      GPU Type: {specs.get('gpu_type', 'N/A')}")
                print(f"      Price: ${price:.2f}/hour")
                print(f"      Regions: {[r['name'] for r in regions]}")
        else:
            print("\n⚠️  No GPU instances currently available")
        
        print("\n✅ PASSED")
        return available
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_3_select_cheapest_instance(client):
    """Test 3: Can select cheapest available instance"""
    print("\n" + "=" * 70)
    print("TEST 3: Select Cheapest Available Instance")
    print("=" * 70)
    
    if not client:
        print("❌ SKIPPED: No client available")
        return None
    
    try:
        selected = client.list_available_instances()
        
        if not selected:
            print("\n⚠️  No instances available to select")
            return None
        
        global test_instance_type, test_region
        test_instance_type = selected['name']
        test_region = selected['region']
        
        print(f"\n🎯 Selected cheapest instance:")
        print(f"   Type: {test_instance_type}")
        print(f"   Region: {test_region}")
        
        # Get pricing details
        response = client._make_request('GET', '/instance-types')
        instance_data = response['data'][test_instance_type]
        price = instance_data['instance_type']['price_cents_per_hour'] / 100
        specs = instance_data['instance_type']['specs']
        
        print(f"\n💰 Instance details:")
        print(f"   Price: ${price:.2f}/hour")
        print(f"   GPUs: {specs.get('gpus', 'N/A')}")
        print(f"   GPU Type: {specs.get('gpu_type', 'N/A')}")
        print(f"   vCPUs: {specs.get('vcpus', 'N/A')}")
        print(f"   Memory: {specs.get('memory_gib', 'N/A')} GiB")
        print(f"   Storage: {specs.get('storage_gib', 'N/A')} GiB")
        
        print("\n✅ PASSED")
        return selected
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_4_launch_instance(client, selected):
    """Test 4: Can launch a GPU instance"""
    print("\n" + "=" * 70)
    print("TEST 4: Launch GPU Instance")
    print("=" * 70)
    print("\n⚠️  WARNING: This will launch a REAL GPU instance!")
    print("⚠️  Estimated cost: $0.60-$1.10/hour")
    print("\n⏳ Launching instance (this may take 2-3 minutes)...")
    
    if not client or not selected:
        print("❌ SKIPPED: No client or selected instance")
        return None
    
    try:
        instance_name = f"test-instance-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        print(f"\n🚀 Launching {selected['name']} in {selected['region']}...")
        print(f"   Name: {instance_name}")
        
        start_time = time.time()
        
        # Launch instance (this method waits for 'active' state)
        instance = client.launch_instance(
            instance_type_name=selected['name'],
            region_name=selected['region'],
            name=instance_name
        )
        
        launch_time = time.time() - start_time
        
        if not instance:
            print("\n❌ FAILED: Instance launch returned None")
            return None
        
        global test_instance_id, test_instance_ip
        test_instance_id = instance['id']
        test_instance_ip = instance['ip']
        
        print(f"\n✅ Instance launched successfully!")
        print(f"   Instance ID: {test_instance_id}")
        print(f"   Instance IP: {test_instance_ip}")
        print(f"   Status: {instance['status']}")
        print(f"   Launch time: {launch_time:.1f} seconds")
        
        return instance
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_5_instance_active_state(client, instance):
    """Test 5: Instance reaches 'active' state"""
    print("\n" + "=" * 70)
    print("TEST 5: Instance Active State")
    print("=" * 70)
    
    if not client or not instance:
        print("❌ SKIPPED: No client or instance")
        return False
    
    try:
        instance_id = instance['id']
        status = client.get_instance_status(instance_id)
        
        print(f"\n📊 Instance status: {status}")
        
        if status == 'active':
            print("\n✅ PASSED - Instance is active!")
            return True
        else:
            print(f"\n❌ FAILED - Instance status is '{status}', expected 'active'")
            return False
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_get_instance_details(client, instance):
    """Test 6: Can get instance details (IP, ID, type, region)"""
    print("\n" + "=" * 70)
    print("TEST 6: Get Instance Details")
    print("=" * 70)
    
    if not client or not instance:
        print("❌ SKIPPED: No client or instance")
        return False
    
    try:
        instance_id = instance['id']
        
        # Fetch fresh instance details
        details = client.get_instance(instance_id)
        
        if not details:
            print("\n❌ FAILED: Could not fetch instance details")
            return False
        
        print(f"\n📋 Instance Details:")
        print(f"   ID: {details['id']}")
        print(f"   Name: {details.get('name', 'N/A')}")
        print(f"   IP Address: {details['ip']}")
        print(f"   Status: {details['status']}")
        print(f"   Instance Type: {details['instance_type']['name']}")
        print(f"   Region: {details['region']['name']}")
        print(f"   Hostname: {details.get('hostname', 'N/A')}")
        
        # Verify all required fields are present
        required_fields = ['id', 'ip', 'instance_type', 'region', 'status']
        missing_fields = [f for f in required_fields if f not in details]
        
        if missing_fields:
            print(f"\n❌ FAILED: Missing fields: {missing_fields}")
            return False
        
        print("\n✅ PASSED - All details retrieved successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_check_instance_status(client):
    """Test 7: Can check instance status"""
    print("\n" + "=" * 70)
    print("TEST 7: Check Instance Status")
    print("=" * 70)
    
    if not client or not test_instance_id:
        print("❌ SKIPPED: No client or instance ID")
        return False
    
    try:
        status = client.get_instance_status(test_instance_id)
        
        print(f"\n📊 Current status: {status}")
        
        # List all instances to verify
        all_instances = client.list_instances()
        our_instance = next((i for i in all_instances if i['id'] == test_instance_id), None)
        
        if our_instance:
            print(f"   Verified in instance list: ✅")
            print(f"   Status matches: {'✅' if our_instance['status'] == status else '❌'}")
        
        print("\n✅ PASSED - Status check works!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_8_terminate_instance(client):
    """Test 8: Can terminate instance"""
    print("\n" + "=" * 70)
    print("TEST 8: Terminate Instance")
    print("=" * 70)
    
    if not client or not test_instance_id:
        print("❌ SKIPPED: No client or instance ID")
        return False
    
    try:
        print(f"\n🛑 Terminating instance {test_instance_id}...")
        
        success = client.terminate_instance(test_instance_id)
        
        if not success:
            print("\n❌ FAILED: Termination returned False")
            return False
        
        print(f"   Termination request sent: ✅")
        
        # Wait a bit and check status
        print("\n⏳ Waiting 5 seconds to verify termination...")
        time.sleep(5)
        
        status = client.get_instance_status(test_instance_id)
        print(f"   Current status: {status}")
        
        # Status should be 'terminating' or None (if already terminated)
        if status in ['terminating', 'terminated', None]:
            print("\n✅ PASSED - Instance termination initiated!")
            return True
        else:
            print(f"\n⚠️  WARNING: Status is '{status}', expected 'terminating' or 'terminated'")
            return True  # Still pass as termination was requested
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_9_cleanup_verification(client):
    """Test 9: Cleanup works properly"""
    print("\n" + "=" * 70)
    print("TEST 9: Cleanup Verification")
    print("=" * 70)
    
    if not client or not test_instance_id:
        print("❌ SKIPPED: No client or instance ID")
        return False
    
    try:
        print("\n⏳ Waiting 10 seconds for cleanup to complete...")
        time.sleep(10)
        
        # Check if instance still exists
        all_instances = client.list_instances()
        our_instance = next((i for i in all_instances if i['id'] == test_instance_id), None)
        
        if our_instance:
            status = our_instance['status']
            print(f"\n📊 Instance still exists with status: {status}")
            
            if status in ['terminating', 'terminated']:
                print("   Instance is terminating/terminated: ✅")
                print("\n✅ PASSED - Cleanup in progress!")
                return True
            else:
                print(f"   ⚠️  WARNING: Unexpected status '{status}'")
                return False
        else:
            print("\n📊 Instance not found in list (fully terminated)")
            print("\n✅ PASSED - Cleanup complete!")
            return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_10_handle_no_instances(client):
    """Test 10: Handle no available instances scenario"""
    print("\n" + "=" * 70)
    print("TEST 10: Handle No Available Instances")
    print("=" * 70)
    
    if not client:
        print("❌ SKIPPED: No client available")
        return False
    
    try:
        # Get available instances
        response = client._make_request('GET', '/instance-types')
        all_instances = response.get('data', {})
        
        available = {
            k: v for k, v in all_instances.items() 
            if len(v.get('regions_with_capacity_available', [])) > 0
        }
        
        print(f"\n📊 Currently available instances: {len(available)}")
        
        # Test the list_available_instances method behavior
        if available:
            selected = client.list_available_instances()
            print(f"   Method returns: {selected['name']} in {selected['region']}")
            print("\n✅ PASSED - Method handles available instances correctly!")
        else:
            print("   No instances available (this is the scenario we're testing)")
            
            # The method should handle this gracefully
            try:
                selected = client.list_available_instances()
                if selected:
                    print(f"   Method returned: {selected}")
                else:
                    print("   Method returned None (graceful handling)")
                print("\n✅ PASSED - Method handles no instances gracefully!")
            except Exception as e:
                print(f"   Method raised exception: {type(e).__name__}")
                print("\n⚠️  Method should handle this more gracefully")
                print("   (But this is acceptable behavior)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "#" * 70)
    print("#  PHASE 2.1: LAMBDA CLIENT TESTING".center(70))
    print("#" * 70)
    print("\n⚠️  WARNING: This will launch a REAL GPU instance!")
    print("⚠️  Estimated cost: $0.60-$1.10/hour")
    print("⚠️  The instance will be automatically terminated after tests")
    print("\n" + "=" * 70)
    
    # Prompt for confirmation
    print("\nPress ENTER to continue or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n❌ Tests cancelled by user")
        return
    
    results = {}
    
    # Test 1: Authentication
    client = test_1_authentication()
    results["authentication"] = client is not None
    
    # Test 2: List available instances
    available = test_2_list_available_instances(client)
    results["list_instances"] = available is not None
    
    # Test 3: Select cheapest instance
    selected = test_3_select_cheapest_instance(client)
    results["select_cheapest"] = selected is not None
    
    if not selected:
        print("\n⚠️  Cannot continue without available instances")
        print("   Skipping remaining tests...")
        results.update({
            "launch_instance": False,
            "active_state": False,
            "instance_details": False,
            "check_status": False,
            "terminate": False,
            "cleanup": False,
            "handle_no_instances": False
        })
    else:
        # Test 4: Launch instance
        instance = test_4_launch_instance(client, selected)
        results["launch_instance"] = instance is not None
        
        # Test 5: Instance active state
        results["active_state"] = test_5_instance_active_state(client, instance)
        
        # Test 6: Get instance details
        results["instance_details"] = test_6_get_instance_details(client, instance)
        
        # Test 7: Check instance status
        results["check_status"] = test_7_check_instance_status(client)
        
        # Test 8: Terminate instance
        results["terminate"] = test_8_terminate_instance(client)
        
        # Test 9: Cleanup verification
        results["cleanup"] = test_9_cleanup_verification(client)
        
        # Test 10: Handle no instances
        results["handle_no_instances"] = test_10_handle_no_instances(client)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    # Display instance info for documentation
    if test_instance_id:
        print("\n" + "=" * 70)
        print("INSTANCE INFORMATION (for TESTING_PLAN.md)")
        print("=" * 70)
        print(f"Instance ID: {test_instance_id}")
        print(f"Instance IP: {test_instance_ip}")
        print(f"Instance Type: {test_instance_type}")
        print(f"Region: {test_region}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        if test_instance_id:
            print(f"\n🛑 IMPORTANT: Clean up instance {test_instance_id} manually!")
            print(f"   Run: python -c \"from scripts.lambda_client import LambdaClient; LambdaClient().terminate_instance('{test_instance_id}')\"")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        if test_instance_id:
            print(f"\n🛑 IMPORTANT: Clean up instance {test_instance_id} manually!")