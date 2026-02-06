"""
Comprehensive test suite for CAN-Graph preprocessing functionality.

Tests cover data loading, graph creation, feature validation,
and edge case handling to ensure robust preprocessing.

Run with: python -m pytest tests/test_preprocessing.py -v
Or:       python -m unittest tests.test_preprocessing
"""

import unittest
import numpy as np
import torch

from src.preprocessing.preprocessing import (
    safe_hex_to_int,
    apply_dynamic_id_mapping,
    build_id_mapping_from_normal,
    dataset_creation_streaming,
    find_csv_files,
    graph_creation,
    GraphDataset,
    NODE_FEATURE_COUNT,
    EDGE_FEATURE_COUNT,
)
from torch_geometric.data import Data


class TestPreprocessing(unittest.TestCase):
    """
    Comprehensive test suite for preprocessing functionality.

    Tests cover data loading, graph creation, feature validation,
    and edge case handling to ensure robust preprocessing.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_root = r"data/automotive/hcrl_sa"
        cls.small_window_size = 10  # Smaller for faster testing
        cls.test_stride = 5

    def test_id_mapping_creation(self):
        """Test CAN ID mapping creation and consistency."""
        print("Testing CAN ID mapping creation...")

        # Test normal data mapping
        id_mapping = build_id_mapping_from_normal(self.test_root)

        self.assertIsInstance(id_mapping, dict)
        self.assertIn('OOV', id_mapping)
        self.assertGreater(len(id_mapping), 1)  # Should have more than just OOV

        # Check that all values are integers
        for key, value in id_mapping.items():
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)

        print(f"  ID mapping created with {len(id_mapping)} entries")

    def test_hex_conversion(self):
        """Test hex-to-decimal conversion robustness."""
        print("Testing hex conversion...")

        # Test valid hex strings
        self.assertEqual(safe_hex_to_int("1A"), 26)
        self.assertEqual(safe_hex_to_int("FF"), 255)
        self.assertEqual(safe_hex_to_int("0"), 0)

        # Test invalid inputs
        self.assertIsNone(safe_hex_to_int("XYZ"))
        self.assertIsNone(safe_hex_to_int(""))
        self.assertIsNone(safe_hex_to_int(None))

        # Test numeric inputs
        self.assertEqual(safe_hex_to_int(123), 123)
        self.assertEqual(safe_hex_to_int(0), 0)

        print("  Hex conversion tests passed")

    def test_single_file_processing(self):
        """Test processing of a single CSV file."""
        print("Testing single file processing...")

        csv_files = find_csv_files(self.test_root, 'train_')
        if not csv_files:
            self.skipTest("No CSV files found for testing")

        # Test with first available file
        test_file = csv_files[0]
        id_mapping = build_id_mapping_from_normal(self.test_root)

        df = dataset_creation_streaming(test_file, id_mapping=id_mapping)

        # Validate DataFrame structure
        expected_columns = ['CAN ID'] + [f'Data{i+1}' for i in range(8)] + ['Source', 'Target', 'label']
        self.assertEqual(list(df.columns), expected_columns)

        # Check normalization
        for col in [f'Data{i+1}' for i in range(8)]:
            self.assertTrue(df[col].between(0, 1).all(), f"{col} not properly normalized")

        # Check for missing values
        self.assertFalse(df.isnull().any().any(), "DataFrame contains NaN values")

        print(f"  Single file processing: {len(df)} rows processed")

    def test_graph_creation_basic(self):
        """Test basic graph creation functionality."""
        print("Testing basic graph creation...")

        dataset = graph_creation(
            self.test_root,
            window_size=self.small_window_size,
            stride=self.test_stride
        )

        self.assertIsInstance(dataset, GraphDataset)
        self.assertGreater(len(dataset), 0)

        print(f"  Created {len(dataset)} graphs")

    def test_graph_structure_validation(self):
        """Test graph structure and feature validation."""
        print("Testing graph structure validation...")

        dataset = graph_creation(
            self.test_root,
            window_size=self.small_window_size,
            stride=self.test_stride
        )

        for i, graph in enumerate(dataset):
            # Basic structure validation
            self.assertIsInstance(graph, Data)
            self.assertIsNotNone(graph.x)
            self.assertIsNotNone(graph.edge_index)
            self.assertIsNotNone(graph.y)

            # Feature dimension validation
            self.assertEqual(graph.x.size(1), NODE_FEATURE_COUNT,
                           f"Graph {i}: incorrect node feature count")

            if graph.edge_attr is not None:
                self.assertEqual(graph.edge_attr.size(1), EDGE_FEATURE_COUNT,
                               f"Graph {i}: incorrect edge feature count")

            # Data type validation
            self.assertEqual(graph.x.dtype, torch.float, f"Graph {i}: incorrect node feature dtype")
            self.assertEqual(graph.edge_index.dtype, torch.long, f"Graph {i}: incorrect edge index dtype")

            # Value range validation
            self.assertFalse(torch.isnan(graph.x).any(), f"Graph {i}: NaN in node features")
            self.assertFalse(torch.isinf(graph.x).any(), f"Graph {i}: Inf in node features")

            # Payload normalization check (columns 1-8)
            payload = graph.x[:, 1:9]
            self.assertTrue(torch.all(payload >= 0) and torch.all(payload <= 1),
                          f"Graph {i}: payload features not normalized to [0,1]")

            # Edge attribute validation
            if graph.edge_attr is not None:
                self.assertFalse(torch.isnan(graph.edge_attr).any(),
                               f"Graph {i}: NaN in edge features")
                self.assertFalse(torch.isinf(graph.edge_attr).any(),
                               f"Graph {i}: Inf in edge features")

            # Test only first 10 graphs for speed
            if i >= 9:
                break

        print("  Graph structure validation passed")

    def test_optimized_processing(self):
        """Test optimized processing with streaming."""
        print("Testing optimized processing...")

        # Test optimized processing
        dataset = graph_creation(
            self.test_root,
            window_size=self.small_window_size,
            stride=self.test_stride
        )

        self.assertIsInstance(dataset, GraphDataset)
        self.assertGreater(len(dataset), 0)

        print(f"  Optimized processing created {len(dataset)} graphs")

    def test_dataset_statistics(self):
        """Test dataset statistics computation."""
        print("Testing dataset statistics...")

        dataset = graph_creation(
            self.test_root,
            window_size=self.small_window_size,
            stride=self.test_stride
        )

        stats = dataset.get_stats()

        # Validate statistics structure
        required_keys = ['num_graphs', 'avg_nodes', 'avg_edges', 'normal_graphs', 'attack_graphs']
        for key in required_keys:
            self.assertIn(key, stats)

        # Validate statistics values
        self.assertEqual(stats['num_graphs'], len(dataset))
        self.assertGreaterEqual(stats['normal_graphs'], 0)
        self.assertGreaterEqual(stats['attack_graphs'], 0)
        self.assertEqual(stats['normal_graphs'] + stats['attack_graphs'], stats['num_graphs'])

        print("  Dataset statistics validation passed")

    def test_apply_dynamic_id_mapping_no_expansion(self):
        """Test that apply_dynamic_id_mapping does NOT expand the mapping for unseen IDs."""
        import pandas as pd

        id_mapping = {100: 0, 200: 1, 'OOV': 2}
        original_len = len(id_mapping)

        # DataFrame with known ID (100) and unseen ID (999)
        df = pd.DataFrame({
            'CAN ID': [100, 999, 100],
            'Source': [100, 999, 100],
            'Target': [999, 100, 999],
            'label': [0, 0, 0],
        })

        result_df, result_mapping = apply_dynamic_id_mapping(df, id_mapping)

        # Mapping must not grow
        self.assertEqual(len(result_mapping), original_len,
                         "Mapping was expanded — unseen IDs should map to OOV, not get new entries")

        # Unseen ID 999 should be mapped to OOV index (2)
        oov = id_mapping['OOV']
        self.assertTrue((result_df['CAN ID'] == oov).sum() == 1,
                        "Unseen CAN ID was not mapped to OOV")
        self.assertTrue((result_df['Source'] == oov).sum() == 1,
                        "Unseen Source was not mapped to OOV")
        self.assertTrue((result_df['Target'] == oov).sum() == 2,
                        "Unseen Target was not mapped to OOV")

        # Known ID 100 should map to 0
        self.assertTrue((result_df['CAN ID'] == 0).sum() == 2)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("Testing edge cases...")

        # Test with non-existent directory
        empty_dataset = graph_creation("/non/existent/path")
        self.assertEqual(len(empty_dataset), 0)

        # Test with empty ID mapping
        empty_mapping = {'OOV': 0}
        dataset = graph_creation(
            self.test_root,
            window_size=self.small_window_size,
            stride=self.test_stride,
            id_mapping=empty_mapping
        )
        self.assertIsInstance(dataset, GraphDataset)

        print("  Edge case testing passed")

    def test_full_pipeline_integration(self):
        """Test complete preprocessing pipeline integration."""
        print("Testing full pipeline integration...")

        # Test with ID mapping return
        dataset, id_mapping = graph_creation(
            self.test_root,
            window_size=self.small_window_size,
            stride=self.test_stride,
            return_id_mapping=True
        )

        self.assertIsInstance(dataset, GraphDataset)
        self.assertIsInstance(id_mapping, dict)
        self.assertIn('OOV', id_mapping)

        # Test reusing ID mapping
        dataset2 = graph_creation(
            self.test_root,
            window_size=self.small_window_size,
            stride=self.test_stride,
            id_mapping=id_mapping
        )

        self.assertIsInstance(dataset2, GraphDataset)

        # Print final statistics
        dataset.print_stats()

        print("  Full pipeline integration test passed")


def run_comprehensive_tests():
    """Run all preprocessing tests with detailed output."""
    print(f"\n{'='*80}")
    print("CAN-GRAPH PREPROCESSING COMPREHENSIVE TEST SUITE")
    print(f"{'='*80}")

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPreprocessing)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run comprehensive test suite if called directly
    success = run_comprehensive_tests()
    exit(0 if success else 1)
