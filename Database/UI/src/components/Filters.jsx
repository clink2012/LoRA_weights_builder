export default function Filters({
  baseModel,
  setBaseModel,
  category,
  setCategory,
  search,
  setSearch,
  hasBlocks,
  setHasBlocks
}) {
  return (
    <div className="space-y-4">

      {/* Base Model */}
      <div>
        <label className="block text-sm font-semibold mb-1">
          Base Model
        </label>
        <select
          className="w-full bg-gray-700 text-white p-2 rounded"
          value={baseModel}
          onChange={(e) => setBaseModel(e.target.value)}
        >
          <option value="">All</option>
          <option value="FLX">Flux</option>
          <option value="FLK">Flux Krea</option>
          <option value="PNY">Pony</option>
          <option value="SD1">SD 1.X</option>
          <option value="SDX">SDXL</option>
          <option value="W21">WAN 2.1</option>
          <option value="W22">WAN 2.2</option>
          <option value="ILL">Illustrious</option>
        </select>
      </div>

      {/* Category */}
      <div>
        <label className="block text-sm font-semibold mb-1">
          Category
        </label>
        <select
          className="w-full bg-gray-700 text-white p-2 rounded"
          value={category}
          onChange={(e) => setCategory(e.target.value)}
        >
          <option value="">All</option>
          <option value="PPL">People</option>
          <option value="STL">Styles</option>
          <option value="UTL">Utils</option>
          <option value="ACT">Action</option>
          <option value="BDY">Body</option>
          <option value="CHT">Characters</option>
          <option value="MCV">Machines / Vehicles</option>
          <option value="CLT">Clothing</option>
          <option value="ANM">Animals</option>
          <option value="BLD">Buildings</option>
          <option value="NAT">Nature</option>
        </select>
      </div>

      {/* Search */}
      <div>
        <label className="block text-sm font-semibold mb-1">
          Search (filename contains)
        </label>
        <input
          type="text"
          className="w-full bg-gray-700 text-white p-2 rounded"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Type to search..."
        />
      </div>

      {/* Blockweights toggle */}
      <div className="flex items-center space-x-2">
        <input
          type="checkbox"
          checked={hasBlocks}
          onChange={(e) => setHasBlocks(e.target.checked)}
        />
        <label className="text-sm">Only LoRAs with block weights</label>
      </div>

    </div>
  );
}
