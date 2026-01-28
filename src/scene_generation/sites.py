"""
Base Station Site Placement
Handles transmitter/receiver position and antenna configuration
"""

import logging
from typing import List, Dict, Tuple, Optional, Literal
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AntennaConfig:
    """Antenna configuration for a site."""
    pattern: str = "iso"  # 'iso', 'dipole', '3gpp_38901'
    orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (azimuth, downtilt, slant)
    polarization: str = "V"  # 'V', 'H', 'dual'
    num_rows: int = 1
    num_cols: int = 1
    vertical_spacing: float = 0.5  # meters
    horizontal_spacing: float = 0.5  # meters
    carrier_freq_ghz: float = 3.5


@dataclass
class Site:
    """Transmitter/receiver site with position and antenna config."""
    site_id: str
    position: Tuple[float, float, float]  # (x, y, z) in scene coordinates
    site_type: Literal["tx", "rx"]  # transmitter or receiver
    antenna: AntennaConfig
    cell_id: Optional[int] = None
    sector_id: Optional[int] = None
    power_dbm: float = 43.0  # Typical macro BS
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for metadata."""
        return {
            'site_id': self.site_id,
            'position': self.position,
            'site_type': self.site_type,
            'cell_id': self.cell_id,
            'sector_id': self.sector_id,
            'power_dbm': self.power_dbm,
            'antenna': {
                'pattern': self.antenna.pattern,
                'orientation': self.antenna.orientation,
                'polarization': self.antenna.polarization,
                'num_rows': self.antenna.num_rows,
                'num_cols': self.antenna.num_cols,
                'vertical_spacing': self.antenna.vertical_spacing,
                'horizontal_spacing': self.antenna.horizontal_spacing,
                'carrier_freq_ghz': self.antenna.carrier_freq_ghz,
            }
        }


class SitePlacer:
    """
    Base station site placement strategies.
    
    Supports:
    - Grid placement (uniform coverage)
    - Random placement (diverse scenarios)
    - Custom placement (from config/measurements)
    - ISD-based placement (3GPP hexagonal grids)
    """
    
    def __init__(
        self,
        strategy: Literal["grid", "random", "custom", "isd"] = "grid",
        default_antenna: Optional[AntennaConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize site placer.
        
        Args:
            strategy: Placement strategy
            default_antenna: Default antenna configuration
            seed: Random seed for reproducibility
        """
        self.strategy = strategy
        self.default_antenna = default_antenna or AntennaConfig()
        
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(f"SitePlacer initialized (strategy={strategy})")
    
    def place(
        self,
        bounds: Tuple[float, float, float, float],  # (xmin, ymin, xmax, ymax)
        num_tx: int = 1,
        num_rx: int = 10,
        height_tx: float = 25.0,  # meters
        height_tx_range: Optional[Tuple[float, float]] = None,
        height_rx: float = 1.5,   # UE height
        custom_positions: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
        isd_meters: Optional[float] = None,  # Inter-site distance for ISD strategy
    ) -> List[Site]:
        """
        Place sites in scene according to strategy.
        
        Args:
            bounds: Scene bounds (xmin, ymin, xmax, ymax) in meters
            num_tx: Number of transmitter sites
            num_rx: Number of receiver sites
            height_tx: Transmitter height above ground
            height_rx: Receiver height above ground
            custom_positions: Dict with 'tx' and 'rx' position lists
            isd_meters: Inter-site distance for hexagonal grid (ISD strategy)
            
        Returns:
            List of Site objects
        """
        if self.strategy == "custom":
            if custom_positions is None:
                raise ValueError("custom_positions required for 'custom' strategy")
            return self._place_custom(custom_positions, height_tx, height_rx)
        
        elif self.strategy == "grid":
            return self._place_grid(bounds, num_tx, num_rx, height_tx, height_rx, height_tx_range)
        
        elif self.strategy == "random":
            return self._place_random(bounds, num_tx, num_rx, height_tx, height_rx, height_tx_range)
        
        elif self.strategy == "isd":
            if isd_meters is None:
                raise ValueError("isd_meters required for 'isd' strategy")
            return self._place_isd(bounds, isd_meters, num_rx, height_tx, height_rx, height_tx_range)
        
        else:
            raise ValueError(f"Unknown placement strategy: {self.strategy}")
    
    def _place_grid(
        self,
        bounds: Tuple[float, float, float, float],
        num_tx: int,
        num_rx: int,
        height_tx: float,
        height_rx: float,
        height_tx_range: Optional[Tuple[float, float]],
    ) -> List[Site]:
        """Grid placement with uniform spacing."""
        xmin, ymin, xmax, ymax = bounds
        sites = []
        
        # Place transmitters in grid
        tx_per_side = int(np.ceil(np.sqrt(num_tx)))
        tx_x = np.linspace(xmin + 50, xmax - 50, tx_per_side)
        tx_y = np.linspace(ymin + 50, ymax - 50, tx_per_side)
        
        tx_count = 0
        for x in tx_x:
            for y in tx_y:
                if tx_count >= num_tx:
                    break
                tx_height = self._sample_tx_height(height_tx, height_tx_range)
                
                # 3-sector site (typical macro BS)
                for sector_id in range(3):
                    azimuth = sector_id * 120.0  # 0°, 120°, 240°
                    antenna = AntennaConfig(
                        pattern="3gpp_38901",
                        orientation=(azimuth, 10.0, 0.0),  # 10° downtilt
                        polarization="dual",
                        num_rows=8,
                        num_cols=8,
                    )
                    
                    site = Site(
                        site_id=f"tx_{tx_count}_sector_{sector_id}",
                        position=(float(x), float(y), tx_height),
                        site_type="tx",
                        antenna=antenna,
                        cell_id=tx_count,
                        sector_id=sector_id,
                        power_dbm=43.0,
                    )
                    sites.append(site)
                
                tx_count += 1
        
        # Place receivers in grid
        rx_per_side = int(np.ceil(np.sqrt(num_rx)))
        rx_x = np.linspace(xmin + 10, xmax - 10, rx_per_side)
        rx_y = np.linspace(ymin + 10, ymax - 10, rx_per_side)
        
        rx_count = 0
        for x in rx_x:
            for y in rx_y:
                if rx_count >= num_rx:
                    break
                
                # UE with isotropic antenna
                antenna = AntennaConfig(
                    pattern="iso",
                    orientation=(0.0, 0.0, 0.0),
                    polarization="V",
                )
                
                site = Site(
                    site_id=f"rx_{rx_count}",
                    position=(float(x), float(y), height_rx),
                    site_type="rx",
                    antenna=antenna,
                    power_dbm=0.0,  # Receiver
                )
                sites.append(site)
                rx_count += 1
        
        logger.info(f"Placed {tx_count} TX sites (x3 sectors) + {rx_count} RX sites (grid)")
        return sites
    
    def _place_random(
        self,
        bounds: Tuple[float, float, float, float],
        num_tx: int,
        num_rx: int,
        height_tx: float,
        height_rx: float,
        height_tx_range: Optional[Tuple[float, float]],
    ) -> List[Site]:
        """Random placement within bounds."""
        xmin, ymin, xmax, ymax = bounds
        sites = []
        
        # Random TX positions (avoid edges)
        margin = 50.0
        tx_x = np.random.uniform(xmin + margin, xmax - margin, num_tx)
        tx_y = np.random.uniform(ymin + margin, ymax - margin, num_tx)
        
        for i in range(num_tx):
            tx_height = self._sample_tx_height(height_tx, height_tx_range)
            # Random orientation
            azimuth = np.random.uniform(0, 360)
            
            antenna = AntennaConfig(
                pattern="3gpp_38901",
                orientation=(azimuth, 10.0, 0.0),
                polarization="dual",
                num_rows=8,
                num_cols=8,
            )
            
            site = Site(
                site_id=f"tx_{i}",
                position=(float(tx_x[i]), float(tx_y[i]), tx_height),
                site_type="tx",
                antenna=antenna,
                cell_id=i,
                power_dbm=43.0,
            )
            sites.append(site)
        
        # Random RX positions
        rx_x = np.random.uniform(xmin + 10, xmax - 10, num_rx)
        rx_y = np.random.uniform(ymin + 10, ymax - 10, num_rx)
        
        for i in range(num_rx):
            antenna = AntennaConfig(pattern="iso")
            
            site = Site(
                site_id=f"rx_{i}",
                position=(float(rx_x[i]), float(rx_y[i]), height_rx),
                site_type="rx",
                antenna=antenna,
            )
            sites.append(site)
        
        logger.info(f"Placed {num_tx} TX + {num_rx} RX sites (random)")
        return sites
    
    def _place_custom(
        self,
        custom_positions: Dict[str, List[Tuple[float, float, float]]],
        height_tx: float,
        height_rx: float,
    ) -> List[Site]:
        """Place sites at custom positions."""
        sites = []
        
        tx_positions = custom_positions.get('tx', [])
        rx_positions = custom_positions.get('rx', [])
        
        for i, (x, y, z) in enumerate(tx_positions):
            antenna = AntennaConfig(
                pattern="3gpp_38901",
                orientation=(0.0, 10.0, 0.0),
                polarization="dual",
            )
            
            site = Site(
                site_id=f"tx_{i}",
                position=(float(x), float(y), float(z)),
                site_type="tx",
                antenna=antenna,
                cell_id=i,
            )
            sites.append(site)
        
        for i, (x, y, z) in enumerate(rx_positions):
            site = Site(
                site_id=f"rx_{i}",
                position=(float(x), float(y), float(z)),
                site_type="rx",
                antenna=AntennaConfig(pattern="iso"),
            )
            sites.append(site)
        
        logger.info(f"Placed {len(tx_positions)} TX + {len(rx_positions)} RX sites (custom)")
        return sites
    
    def _place_isd(
        self,
        bounds: Tuple[float, float, float, float],
        isd_meters: float,
        num_rx: int,
        height_tx: float,
        height_rx: float,
        height_tx_range: Optional[Tuple[float, float]],
    ) -> List[Site]:
        """
        Place TX sites in hexagonal grid (3GPP style).
        
        Args:
            bounds: Scene bounds
            isd_meters: Inter-site distance (typical: 200m urban, 500m suburban)
            num_rx: Number of receiver sites
            height_tx: TX height
            height_rx: RX height
            
        Returns:
            List of sites with hexagonal TX grid + random RX
        """
        xmin, ymin, xmax, ymax = bounds
        sites = []
        
        # Hexagonal grid parameters
        dx = isd_meters
        dy = isd_meters * np.sqrt(3) / 2
        
        # Generate hex grid within bounds
        x_positions = np.arange(xmin, xmax, dx)
        y_positions = np.arange(ymin, ymax, dy)
        
        tx_count = 0
        for i, x in enumerate(x_positions):
            for j, y in enumerate(y_positions):
                # Offset every other row
                x_offset = (dx / 2) if j % 2 == 1 else 0
                x_pos = x + x_offset
                
                if xmin < x_pos < xmax and ymin < y < ymax:
                    tx_height = self._sample_tx_height(height_tx, height_tx_range)
                    # 3-sector site
                    for sector_id in range(3):
                        azimuth = sector_id * 120.0
                        
                        antenna = AntennaConfig(
                            pattern="3gpp_38901",
                            orientation=(azimuth, 10.0, 0.0),
                            polarization="dual",
                            num_rows=8,
                            num_cols=8,
                        )
                        
                        site = Site(
                            site_id=f"tx_{tx_count}_sector_{sector_id}",
                            position=(x_pos, y, tx_height),
                            site_type="tx",
                            antenna=antenna,
                            cell_id=tx_count,
                            sector_id=sector_id,
                            power_dbm=43.0,
                        )
                        sites.append(site)
                    
                    tx_count += 1
        
        # Random RX placement
        rx_x = np.random.uniform(xmin + 10, xmax - 10, num_rx)
        rx_y = np.random.uniform(ymin + 10, ymax - 10, num_rx)
        
        for i in range(num_rx):
            site = Site(
                site_id=f"rx_{i}",
                position=(float(rx_x[i]), float(rx_y[i]), height_rx),
                site_type="rx",
                antenna=AntennaConfig(pattern="iso"),
            )
            sites.append(site)
        
        logger.info(f"Placed {tx_count} TX sites (ISD={isd_meters}m) + {num_rx} RX sites")
        return sites

    def _sample_tx_height(
        self,
        default_height: float,
        height_range: Optional[Tuple[float, float]],
    ) -> float:
        if not height_range:
            return float(default_height)
        low, high = height_range
        if low > high:
            low, high = high, low
        return float(np.random.uniform(low, high))


if __name__ == "__main__":
    # Test site placement
    placer = SitePlacer(strategy="grid", seed=42)
    
    bounds = (0, 0, 500, 500)  # 500m x 500m scene
    sites = placer.place(bounds, num_tx=3, num_rx=10)
    
    logger.info(f"\nPlaced {len(sites)} sites:")
    for site in sites[:5]:  # Show first 5
        logger.info(f"  {site.site_id}: pos={site.position}, type={site.site_type}")
